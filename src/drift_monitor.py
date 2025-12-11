import json
import os
import sys
import pandas as pd
import numpy as np
import ast

STATS_PATH = "metrics/training_stats.json"
LOGS_PATH = "logs/predictions.csv"
DRIFT_THRESHOLD = 3.0

def check_drift():
    # 1. Validation
    if not os.path.exists(STATS_PATH):
        print(f"SKIP: No baseline stats found at {STATS_PATH}")
        return False
    if not os.path.exists(LOGS_PATH):
        print(f"SKIP: No logs found at {LOGS_PATH}")
        return False

    with open(STATS_PATH, 'r') as f:
        baseline = json.load(f)

    # 2. Load and Parse Logs
    try:
        df = pd.read_csv(LOGS_PATH)
        # Parse string "[...]" into actual list
        df['features_list'] = df['features'].apply(ast.literal_eval)
    except Exception as e:
        print(f"ERROR: Failed to read logs: {e}")
        return False

    if len(df) < 5:
        print("SKIP: Not enough data points.")
        return False

    print(f"[Monitor] Analyzing {len(df)} recent predictions...")

    # 3. Check for Drift
    feature_names = baseline['feature_order']
    means = baseline['means']
    stds = baseline['stds']
    drift_detected = False

    # Convert list of lists into a numpy array
    live_matrix = np.array(df['features_list'].tolist())

    for i, feature in enumerate(feature_names):
        if i >= live_matrix.shape[1]: break 

        current_mean = np.mean(live_matrix[:, i])
        base_mean = means[feature]
        base_std = stds[feature]

        if base_std == 0: continue

        z_score = abs((current_mean - base_mean) / base_std)
        print(f"Feature: {feature:<15} | Z-Score: {z_score:.4f}")

        if z_score > DRIFT_THRESHOLD:
            print(f"!!! DRIFT DETECTED: {feature} !!!")
            drift_detected = True

    return drift_detected

if __name__ == "__main__":
    if check_drift():
        sys.exit(1) # Signal Failure -> Trigger Retrain
    else:
        sys.exit(0) # Signal Success -> All Good