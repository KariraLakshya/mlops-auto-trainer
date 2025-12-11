import argparse
import ast
import json
import math
import os

import pandas as pd


def parse_features_column(series):
    """
    Convert strings like "[5.1, 3.5]" into real Python lists.
    """
    parsed = []
    for v in series:
        try:
            parsed.append(ast.literal_eval(str(v)))
        except Exception:
            # skip bad rows
            continue
    return parsed


def main(stats_path: str, logs_path: str, threshold: float):
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Training stats not found at {stats_path}")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs not found at {logs_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    feature_order = stats["feature_order"]
    train_means = stats["means"]
    train_stds = stats["stds"]

    # Load logs
    logs_df = pd.read_csv(logs_path)
    if logs_df.empty:
        print("No prediction logs yet — cannot compute drift.")
        return

    parsed = parse_features_column(logs_df["features"])
    if not parsed:
        print("No valid feature rows parsed from logs.")
        return

    # Build DataFrame from list-of-lists, using same feature_order
    live_df = pd.DataFrame(parsed, columns=feature_order)

    live_means = live_df.mean().to_dict()

    per_feature_z = {}
    for feat in feature_order:
        m_train = train_means.get(feat, 0.0)
        s_train = train_stds.get(feat, 0.0)
        m_live = live_means.get(feat, 0.0)

        if s_train == 0 or math.isnan(s_train):
            # if feature had no variance in training, skip or treat as 0 drift
            z = 0.0
        else:
            z = abs(m_live - m_train) / s_train

        per_feature_z[feat] = z

    # overall drift score = max z-score across features
    overall_drift = max(per_feature_z.values()) if per_feature_z else 0.0

    print("Per-feature drift (z-scores):")
    for k, v in per_feature_z.items():
        print(f"  {k}: {v:.3f}")

    print(f"DRIFT_SCORE={overall_drift:.3f}")
    if overall_drift >= threshold:
        print(f"!! Drift exceeds threshold {threshold} — consider retraining.")
    else:
        print(f"Drift below threshold {threshold} — model is likely OK for now.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stats", default="metrics/training_stats.json")
    p.add_argument("--logs", default="logs/predictions.csv")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    main(args.stats, args.logs, args.threshold)
