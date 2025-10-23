#!/usr/bin/env python3
# scripts/audit_split.py
import os, sys, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.getenv("MANIFEST", "data/train.csv"),
                    help="CSV with columns: path,label,group_id,split,timestamp")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest not found: {args.manifest}")
        sys.exit(2)

    df = pd.read_csv(args.manifest)

    # Basic checks
    required = ["path","label","group_id","split","timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        sys.exit(2)

    # Leakage check: same group_id across multiple splits?
    piv = pd.crosstab(df["group_id"], df["split"])
    leak_groups = piv[(piv > 0).sum(axis=1) > 1]
    if len(leak_groups) == 0:
        print("No group leakage across splits.")
    else:
        print("Group leakage across splits:")
        print(piv[piv.index.isin(leak_groups.index)])

    # Split stats
    for s in ["train","val","test"]:
        sub = df[df["split"] == s]
        if len(sub) == 0:
            pos_rate = float("nan")
        else:
            pos_rate = sub["label"].sum()/len(sub)
        print(f"{s}: n={len(sub)} | pos={int(sub['label'].sum())} neg={len(sub)-int(sub['label'].sum())} pos_rate={pos_rate:.4f}")

    # Simple temporal sanity (if timestamps are numeric/parsable)
    try:
        ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.assign(ts=ts)
        tr = df[df["split"]=="train"]["ts"]
        te = df[df["split"]=="test"]["ts"]
        if not tr.empty and not te.empty:
            print(f"train max ts: {tr.max()} | test min ts: {te.min()}")
    except Exception as e:
        pass

if __name__ == "__main__":
    main()

