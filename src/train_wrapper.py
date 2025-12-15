import argparse
import subprocess
import json
import os
import sys
import shutil

# Setup Logging Paths
LOG_FILE = "run.log"
ARTIFACTS_DIR = "artifacts"

def log(message):
    """Prints to console and appends to log file"""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def run_training_pipeline(args):
    # 1. Setup Artifacts Directory (Clean start)
    if os.path.exists(ARTIFACTS_DIR):
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR)
    
    log("=== MLOps Pipeline Started ===")
    
    # 2. Load Manifest to check Problem Type
    if not os.path.exists("mlmanifest.json"):
        log(" Error: mlmanifest.json not found!")
        sys.exit(1)

    with open("mlmanifest.json", "r") as f:
        manifest = json.load(f)

    # 3. Extract Configs
    problem_type = manifest.get("problem_type", "tabular").lower()
    train_script = manifest.get("train_script")
    data_path = manifest.get("data_path")
    
    # 4. Strategy: Drift Monitoring (Only for Tabular)
    if problem_type == "tabular":
        log(f"📊 Problem Type: Tabular. Running Drift Monitor...")
        try:
            # We run this as a subprocess so it doesn't kill the main wrapper if it fails
            subprocess.run(
                ["python", "src/compute_training_stats.py", "--data", data_path], 
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            log(" Drift stats computed.")
        except subprocess.CalledProcessError as e:
            log(f" Warning: Drift Monitor failed:\n{e.stdout}")
    else:
        log(f" Problem Type: {problem_type}. Skipping Numeric Drift Monitor.")

    # 5. Run User's Training Script
    log(f" Launching User Script: {train_script}")
    log(f" Saving extra plots to: {ARTIFACTS_DIR}/")
    
    try:
        # We run the user script and capture ALL output to the log file
        process = subprocess.run(
            ["python", train_script, "--data", data_path, "--out", args.out],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        # Write user script output to our log
        log("\n--- [USER SCRIPT OUTPUT START] ---")
        log(process.stdout)
        log("--- [USER SCRIPT OUTPUT END] ---\n")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, train_script)

        log(" Training completed successfully.")

    except subprocess.CalledProcessError as e:
        log(f" CRITICAL ERROR: Training script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # We allow these to be passed, but we prefer reading from manifest inside
    parser.add_argument("--script", required=False)
    parser.add_argument("--data", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    # Clear previous log
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    
    try:
        run_training_pipeline(args)
    except Exception as e:
        log(f"❌ SYSTEM ERROR: {str(e)}")
        sys.exit(1)