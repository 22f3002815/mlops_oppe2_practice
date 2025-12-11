# scripts/add_location.py
"""
Add a synthetic categorical 'location' column randomly assigned to Location_A or Location_B.
Usage:
  python scripts/add_location.py --input data/v0/transactions_2022.csv --output data/v0/transactions_2022_with_location.csv --seed 42 --pA 0.5
"""
import argparse
import pandas as pd
import numpy as np
import os

def add_location(input_csv, out_csv, seed=42, pA=0.5):
    df = pd.read_csv(input_csv)
    rng = np.random.default_rng(seed)
    # draw booleans for Location_A with probability pA
    is_A = rng.random(len(df)) < pA
    df["location"] = ["Location_A" if v else "Location_B" for v in is_A]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved with location: {out_csv} (counts: {df['location'].value_counts().to_dict()})")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pA", type=float, default=0.5, help="Probability of Location_A assignment")
    args = parser.parse_args()
    add_location(args.input, args.output, seed=args.seed, pA=args.pA)
