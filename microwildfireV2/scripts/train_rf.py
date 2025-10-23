#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as T

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.getenv("MANIFEST", "data/train.csv"))
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=None)
    return ap.parse_args()


class Featurizer:
    """Frozen ResNet18 avgpool features -> 512-D"""
    def __init__(self):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*(list(m.children())[:-1]))  # up to avgpool
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


def load_split(df, split):
    sub = df[df.split==split].reset_index(drop=True)
    X, y = [], []
    fz = Featurizer()
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"feats:{split}", ncols=100):
        X.append(fz.one(r.path))
        y.append(int(r.label))
    return np.stack(X), np.array(y)


def main():
    args = parse_args()
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.manifest)
    for col in ["path","label","split"]:
        assert col in df.columns, f"manifest missing column {col}"

    Xtr, ytr = load_split(df, "train")
    Xva, yva = load_split(df, "val")
    Xte, yte = load_split(df, "test")

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(np.concatenate([Xtr, Xva], axis=0), np.concatenate([ytr, yva], axis=0))

    pte = rf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, pte)
    print(f"[val]  AUC={roc_auc_score(yva, rf.predict_proba(Xva)[:,1]):.4f}")
    print(f"[test] AUC={auc:.4f}")

    dump(rf, out_dir / "rf_model.joblib")
    with open(out_dir / "rf_test_probs.json", "w") as f:
        json.dump({"probs": pte.tolist(),
                   "labels": yte.astype(int).tolist(),
                   "auc": float(auc)}, f)
    print("Saved: out/rf_model.joblib, out/rf_test_probs.json")


if __name__ == "__main__":
    main()

