"""
scripts/poison_data.py

Usage:
  python scripts/poison_data.py \
    --input data/v0/transactions_2022.csv \
    --out_dir data/poisoned \
    --ratios 0.02 0.08 0.20 \
    --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np

def poison_df(df, flip_frac, seed=42):
    """
    Flip `flip_frac` fraction of class-0 rows to class-1 (randomly).
    Return a new DataFrame.
    """
    df = df.copy()
    class0_idx = df[df["Class"] == 0].index.to_numpy()
    n_flip = int(np.floor(len(class0_idx) * flip_frac))
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(class0_idx, size=n_flip, replace=False)
    df.loc[flip_idx, "Class"] = 1
    return df

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    for frac in args.ratios:
        outpath = os.path.join(args.out_dir, f"poisoned_{int(frac*100)}_percent.csv")
        poisoned = poison_df(df, flip_frac=frac, seed=args.seed)
        poisoned.to_csv(outpath, index=False)
        print(f"Saved {outpath} (flipped {frac*100:.1f}% of class-0 -> 1)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to transactions_2022.csv")
    parser.add_argument("--out_dir", default="data/poisoned", help="Output directory for poisoned files")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.02, 0.08, 0.20], help="Fractions to flip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
