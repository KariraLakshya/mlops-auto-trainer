import json
import os
import sys
import argparse
from datetime import datetime

REGISTRY_PATH = "metrics/model_registry.json"

def register_model(accuracy, run_id):
    # 1. Load or Initialize Registry
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    # 2. Determine Version (Auto-Increment)
    next_version = len(history) + 1
    version_tag = f"v{next_version}"

    # 3. Champion Logic (Compare against history)
    current_max = 0.0
    if history:
        current_max = max(m['accuracy'] for m in history)
    
    is_champion = accuracy >= current_max
    
    # 4. Save New Record
    record = {
        "version": version_tag,
        "date": datetime.utcnow().isoformat(),
        "accuracy": accuracy,
        "run_id": run_id,
        "is_champion": is_champion
    }
    
    history.append(record)

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"[Registry] Registered {version_tag} | Accuracy: {accuracy:.4f}")
    
    if is_champion:
        print(f"[Registry] 🚀 NEW CHAMPION! (Beat previous best: {current_max:.4f})")
        return True
    else:
        print(f"[Registry] 📉 Improvement needed. (Best is {current_max:.4f})")
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--accuracy", type=float, required=True)
    p.add_argument("--run-id", default="manual_run")
    args = p.parse_args()
    is_champ = register_model(args.accuracy, args.run_id)
    if is_champ:
        sys.exit(0)
    else:
        sys.exit(1)