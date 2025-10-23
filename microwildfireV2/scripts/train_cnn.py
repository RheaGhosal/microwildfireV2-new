#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

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
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--ens_last_k", type=int, default=5, help="ensemble last-k epochs on test")
    return ap.parse_args()


class ChipDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        # standard ImageNet transforms
        if train:
            self.tf = T.Compose([
                T.Resize((224,224), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((224,224), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        x = self.tf(img)
        y = int(row.label)
        return x, y


def make_loaders(df, bs, num_workers):
    tr = df[df.split=="train"]
    va = df[df.split=="val"]
    te = df[df.split=="test"]

    dl_tr = DataLoader(ChipDataset(tr, train=True),  batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ChipDataset(va, train=False), batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ChipDataset(te, train=False), batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va, dl_te, tr, va, te


def build_model():
    # ResNet18, replace the last fc with a binary head
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 1)
    return m


@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs); all_y.append(yb.numpy())
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
    for col in ["path","label","split"]:
        assert col in df.columns, f"manifest missing column {col}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr, dl_va, dl_te, tr, va, te = make_loaders(df, args.batch_size, args.num_workers)

    model = build_model().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_va_auc = -1.0
    last_k_test_probs = []
    last_k = args.ens_last_k

    for epoch in trange(args.epochs, desc="Training", ncols=100):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.float().to(device)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        va_auc, _, _ = eval_auc(model, dl_va, device)
        print(f"[val] AUC={va_auc:.4f}")

        # keep last-k test probs for ensembling
        te_auc, te_probs, te_labels = eval_auc(model, dl_te, device)
        last_k_test_probs.append(te_probs)
        if len(last_k_test_probs) > last_k:
            last_k_test_probs.pop(0)

        if va_auc > best_va_auc:
            best_va_auc = va_auc
            torch.save(model.state_dict(), out_dir / "cnn_model.pt")

    # final eval: single model
    te_auc, te_probs, te_labels = eval_auc(model, dl_te, device)
    print(f"[test] AUC={te_auc:.4f}")

    # ensemble of last-k (simple average)
    if last_k_test_probs:
        ens_probs = np.mean(np.stack(last_k_test_probs, axis=0), axis=0)
        try:
            ens_auc = roc_auc_score(te_labels, ens_probs)
        except Exception:
            ens_auc = float("nan")
        print(f"[test-ensemble(last {last_k})] AUC={ens_auc:.4f}")
        with open(out_dir / "cnn_test_probs.json", "w") as f:
            json.dump({"probs": ens_probs.tolist(),
                       "labels": te_labels.astype(int).tolist(),
                       "auc": float(ens_auc)}, f)
    else:
        with open(out_dir / "cnn_test_probs.json", "w") as f:
            json.dump({"probs": te_probs.tolist(),
                       "labels": te_labels.astype(int).tolist(),
                       "auc": float(te_auc)}, f)

    print("Saved: out/cnn_model.pt, out/cnn_test_probs.json")


if __name__ == "__main__":
    main()

