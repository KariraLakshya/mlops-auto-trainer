import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True, help="Path to training script (repo-relative)")
    p.add_argument("--data", required=True, help="Path to training data")
    p.add_argument("--out", required=True, help="Output model path")
    args = p.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: Training script not found at {script_path}", file=sys.stderr)
        sys.exit(2)

    cmd = [sys.executable, str(script_path), "--data", args.data, "--out", args.out]
    print("[train_wrapper] Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[train_wrapper] Training script failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
