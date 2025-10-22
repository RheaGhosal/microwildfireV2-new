#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.getenv("MANIFEST", "data/train.csv"))
    ap.add_argument("--seq_len", type=int, default=5, help="sequence length (sliding window)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


class ImgFeatExtractor:
    """Frozen ResNet18 features (512-D)."""
    def __init__(self):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*(list(m.children())[:-1]))
        self.backbone.eval()
        self.tf = T.Compose([
            T.Resize((224,224), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def one(self, path):
        img = Image.open(path).convert("RGB")
        x = self.tf(img).unsqueeze(0)
        feats = self.backbone(x).squeeze().numpy()  # (512,)
        return feats


def build_sequences(df, seq_len):
    """Group by group_id, sort by timestamp, make sliding windows.
    Label = label of last element in window. Split = split of last element.
    """
    assert {"group_id","timestamp","split","label","path"}.issubset(df.columns)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").astype("int64")//10**9
    buckets = defaultdict(list)
    for gid, sub in df.groupby("group_id"):
        sub = sub.sort_values("timestamp")
        paths = sub["path"].tolist()
        labels = sub["label"].tolist()
        splits = sub["split"].tolist()
        for i in range(len(sub)):
            j0 = max(0, i - (seq_len-1))
            window_paths = paths[j0:i+1]
            # left-pad with first frame if shorter than seq_len
            if len(window_paths) < seq_len:
                window_paths = [window_paths[0]] * (seq_len - len(window_paths)) + window_paths
            buckets[splits[i]].append({
                "paths": window_paths,
                "label": int(labels[i]),
                "group_id": gid
            })
    return buckets  # dict: split -> list of records


class SeqDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.fz = ImgFeatExtractor()

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        feats = [self.fz.one(p) for p in rec["paths"]]  # seq_len x 512
        x = np.stack(feats).astype("float32")
        y = rec["label"]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class LSTMHead(nn.Module):
    def __init__(self, in_dim=512, hidden=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):         # x: (B, T, 512)
        out, _ = self.lstm(x)
        h = out[:, -1, :]         # last step
        return self.fc(h).squeeze(1)


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs); all_y.append(yb.numpy())
    if not all_p:
        return float("nan"), np.array([]), np.array([])
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return auc, p, y


def main():
    args = parse_args()
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.manifest)
    for col in ["path","label","split","group_id","timestamp"]:
        assert col in df.columns, f"manifest missing column {col}"

    buckets = build_sequences(df, args.seq_len)
    tr_recs = buckets.get("train", [])
    va_recs = buckets.get("val", [])
    te_recs = buckets.get("test", [])

    dl_tr = DataLoader(SeqDataset(tr_recs), batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    dl_va = DataLoader(SeqDataset(va_recs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dl_te = DataLoader(SeqDataset(te_recs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMHead().to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        va_auc, _, _ = eval_auc(model, dl_va, device)
        print(f"Epoch {ep}/{args.epochs}  [val] AUC={va_auc:.4f}")

    te_auc, te_probs, te_labels = eval_auc(model, dl_te, device)
    print(f"[test] AUC={te_auc:.4f}")

    torch.save(model.state_dict(), out_dir / "lstm_model.pt")
    with open(out_dir / "lstm_test_probs.json", "w") as f:
        json.dump({"probs": te_probs.tolist(),
                   "labels": te_labels.astype(int).tolist(),
                   "auc": float(te_auc)}, f)
    print("Saved: out/lstm_model.pt, out/lstm_test_probs.json")


if __name__ == "__main__":
    main()

