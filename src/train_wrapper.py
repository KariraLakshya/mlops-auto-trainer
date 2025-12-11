import argparse
import subprocess
import sys
import os
from pathlib import Path

# Import the stats function directly
# Since we run this script from inside src/, this import works
try:
    from compute_training_stats import compute_stats
except ImportError:
    # Fallback if run from root as module
    from src.compute_training_stats import compute_stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True) 
    p.add_argument("--data", required=True)   
    p.add_argument("--out", required=True)    
    args = p.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: Training script not found at {script_path}", file=sys.stderr)
        sys.exit(2)

    # 1. Run User Training Script
    cmd = [sys.executable, str(script_path), "--data", args.data, "--out", args.out]
    print("[Wrapper] Running User Script:", " ".join(cmd))
    
    try:
        subprocess.check_call(cmd)
        print("[Wrapper] Training successful.")
    except subprocess.CalledProcessError as e:
        print(f"[Wrapper] User script failed with code {e.returncode}")
        sys.exit(e.returncode)

    # 2. Run System Profiling (Automatic)
    stats_path = "metrics/training_stats.json"
    print("[Wrapper] Generating drift baseline...")
    
    try:
        compute_stats(args.data, stats_path)
    except Exception as e:
        print(f"[Wrapper] WARNING: Failed to generate stats: {e}")

if __name__ == "__main__":
    main()