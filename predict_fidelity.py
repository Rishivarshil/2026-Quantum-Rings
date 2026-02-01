import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Quantum Fidelity Prediction Pipeline Wrapper")
    
    parser.add_argument("--circuit_dir", type=str, required=True, help="Directory containing QASM files")
    parser.add_argument("--precision", type=str, choices=["single", "double"], required=True, help="Numerical precision")
    parser.add_argument("--backend", type=str, choices=["GPU", "CPU"], required=True, help="Execution backend")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold value for fidelity")

    args = parser.parse_args()

    # 2. Define the scripts you want to run in sequence
    # Replace these filenames with your actual processing scripts
    pipeline_scripts = [
        "qasm_parsing.py",  # Script to process QASM into features
    ]

    for script in pipeline_scripts:
        print(f"--- Running {script} ---")
        
        # Build the command to run the external script with parameters
        command = [
            sys.executable, script,
            "--circuit_dir", args.circuit_dir,
            "--precision", args.precision,
            "--backend", args.backend,
            "--threshold", str(args.threshold)
        ]

        try:
            # Run the script and wait for it to finish
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}: {e}")
            sys.exit(1)

    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()