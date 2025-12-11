import json
import os
import argparse
import pandas as pd
import numpy as np

def compute_stats(data_path, output_path):
    print(f"[Stats] Reading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Remove target column if present (we only monitor features)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    # Select only numeric columns for statistics
    numeric_df = df.select_dtypes(include=[np.number])
    
    means = numeric_df.mean().to_dict()
    stds = numeric_df.std(ddof=0).to_dict()

    # Save structure
    stats = {
        "feature_order": list(numeric_df.columns),
        "means": means,
        "stds": stds,
        "metadata": {
            "row_count": len(df),
            "source": data_path
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[Stats] Training baseline saved to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    compute_stats(args.data, args.out)