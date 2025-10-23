
import argparse, json, glob, os, sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default="data/*.csv", help="CSV glob to load")
    ap.add_argument("--label_col", default="", help="Label column name (auto-detect if empty)")
    ap.add_argument("--id_col", default="", help="Optional ID column to save; else uses integer index")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--out",   default="data/splits.json")
    ap.add_argument("--no_stratify", action="store_true", help="Disable stratification")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    files = sorted(glob.glob(args.data_glob))
    if not files:
        print(f"No CSVs found for pattern: {args.data_glob}", file=sys.stderr)
        sys.exit(2)

    # Load and concat all CSVs
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)
    if not dfs:
        print("No readable CSVs.", file=sys.stderr); sys.exit(2)
    df = pd.concat(dfs, ignore_index=True)

    # Determine label column
    label_col = args.label_col.strip()
    candidates = [c for c in ["label","target","y","class","risk","fire","wildfire"] if c in df.columns]
    if not label_col:
        if candidates:
            label_col = candidates[0]
        else:
            if args.no_stratify:
                label_col = None
            else:
                print("Could not auto-detect a label column. "
                      "Pass --label_col <name> or use --no_stratify.", file=sys.stderr)
                print(f"Available columns: {list(df.columns)[:20]} ...", file=sys.stderr)
                sys.exit(2)

    # Pick ID column or use integer index
    if args.id_col and args.id_col in df.columns:
        ids = df[args.id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    # Build splits
    if label_col is not None:
        y = df[label_col].values
        # train vs temp (val+test), then split temp
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0-args.train, random_state=args.seed)
        train_idx, temp_idx = next(sss1.split(ids, y))
        y_temp = y[temp_idx]
        temp_ids = [ids[i] for i in temp_idx]

        # proportion of val within temp
        val_prop = args.val / (args.val + args.test)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_prop, random_state=args.seed)
        val_rel_idx, test_rel_idx = next(sss2.split(temp_ids, y_temp))
        val_idx = [temp_idx[i] for i in val_rel_idx]
        test_idx = [temp_idx[i] for i in test_rel_idx]
    else:
        train_idx, temp_idx = train_test_split(
            list(range(len(ids))), test_size=1.0-args.train, random_state=args.seed, shuffle=True)
        val_prop = args.val / (args.val + args.test)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1.0 - val_prop, random_state=args.seed, shuffle=True)

    out = {
        "seed": args.seed,
        "proportions": {"train": args.train, "val": args.val, "test": args.test},
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "ids": {
            "train": [ids[i] for i in train_idx],
            "val":   [ids[i] for i in val_idx],
            "test":  [ids[i] for i in test_idx],
        },
        "meta": {
            "data_glob": args.data_glob,
            "label_col": label_col if label_col is not None else "",
            "id_col": args.id_col if args.id_col in df.columns else "",
            "stratified": label_col is not None
        }
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} with counts:", out["counts"])

if __name__ == "__main__":
    main()
