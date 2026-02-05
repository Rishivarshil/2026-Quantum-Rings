import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Quantum Fidelity Prediction Pipeline Wrapper")
    
    parser.add_argument("--circuit", type=str, required=True, help="Directory containing QASM files")
    parser.add_argument("--precision", type=str, choices=["single", "double"], required=True, help="Numerical precision")
    parser.add_argument("--backend", type=str, choices=["GPU", "CPU"], required=True, help="Execution backend")

    args = parser.parse_args()

    # 2. Define the scripts you want to run in sequence
    # Replace these filenames with your actual processing scripts
    pipeline_scripts = [
        "qasm_parsing.py",  # Script to process QASM into features
        "gen_embeddings.py"   # Script to generate embeddings
        "runtime_prediction.py"  # Final prediction script
    ]

    try:
        command = [
            sys.executable, "qasm_parsing.py",
            "--circuit_dir", args.circuit]
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running qasm_parsing.py: {e}")
        sys.exit(1)
    
    try:
        command = [
            sys.executable, "gen_embeddings.py",
            "--circuit_dir", args.circuit]
        result = subprocess.run(command, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running gen_embeddings.py: {e}")
        sys.exit(1)
    try:
        command = [
            sys.executable, "fidelity_prediction.py",
            "--circuit_dir", args.circuit,
            "--precision", args.precision,
            "--backend", args.backend,
        ]   
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running fidelity_prediction.py: {e}")
        sys.exit(1)

    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()