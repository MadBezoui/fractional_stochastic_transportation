import os
import sys
import subprocess

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("=== Reproducing All Fractional Stochastic Transportation Results ===")
    
    # 1. Create necessary output directories
    os.makedirs("manuscript/figures", exist_ok=True)
    os.makedirs("computational_validation/results", exist_ok=True)
    
    # 2. Run experiments (Baselines, PH, Calibration) which will generate figures and CSVs
    run_command("python computational_validation/src/experiments.py")
    
    # 3. Generate tables
    run_command("python generate_all_tables.py")
    
    print("=== Reproduction Completed Successfully ===")
