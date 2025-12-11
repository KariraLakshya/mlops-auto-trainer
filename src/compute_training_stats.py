import json
import os
import argparse

import pandas as pd


def main(data_path: str, out_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if "target" not in df.columns:
        # assume last column is target
        df = df.rename(columns={df.columns[-1]: "target"})

    X = df.drop(columns=["target"])

    means = X.mean().to_dict()
    stds = X.std(ddof=0).to_dict()  # population std

    stats = {
        "feature_order": list(X.columns),
        "means": means,
        "stds": stds,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote training stats to {out_path}")
    print("Feature order:", stats["feature_order"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.csv")
    p.add_argument("--out", default="metrics/training_stats.json")
    args = p.parse_args()
    main(args.data, args.out)
