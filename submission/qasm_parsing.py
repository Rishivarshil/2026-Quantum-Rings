import argparse
import os
import qiskit.qasm2 as qasm
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import pandas as pd
import math
from qiskit.circuit import Barrier

import re

qelib1_pattern = r'(include\s+"qelib1\.inc";)'

rccx_snippet = """
gate rccx a, b, c {
    u2(0, pi) c;
    u1(pi/4) c;
    cx b, c;
    u1(-pi/4) c;
    cx a, c;
    u1(pi/4) c;
    cx b, c;
    u1(-pi/4) c;
    u2(0, pi) c;
}"""

rzz_snippet = """
gate rzz(theta) a, b {
    cx a, b;
    u1(theta) b;
    cx a, b;
}"""

cry_snippet = """
gate cry(theta) a, b {
    u3(theta/2, 0, 0) b;
    cx a, b;
    u3(-theta/2, 0, 0) b;
    cx a, b;
}"""

swap_snippet = """
gate swap a, b {
    cx a, b;
    cx b, a;
    cx a, b;
}"""

p_snippet = """
gate p(lambda) q {
    u1(lambda) q;
}"""

cswap_snippet = """
gate cswap a, b, c {
    cx c, b;
    ccx a, b, c;
    cx c, b;
}"""

# Convert all u -> u3 to eliminate parsing errors
def standardize_qasm_gates(qasm_content):
    
    if 'rccx' in qasm_content and 'gate rccx' not in qasm_content:
    
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + rccx_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending rccx gate definition at the beginning.")
            qasm_content = rccx_snippet + "\n" + qasm_content

    if 'rzz' in qasm_content and 'gate rzz' not in qasm_content:
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + rzz_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending rzz gate definition at the beginning.")
            qasm_content = rzz_snippet + "\n" + qasm_content

    if 'cry' in qasm_content and 'gate cry' not in qasm_content:
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + cry_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending cry gate definition at the beginning.")
            qasm_content = cry_snippet + "\n" + qasm_content

    if 'swap' in qasm_content and 'gate swap' not in qasm_content:
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + swap_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending swap gate definition at the beginning.")
            qasm_content = swap_snippet + "\n" + qasm_content

    if 'p(' in qasm_content and 'gate p' not in qasm_content:
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + p_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending p gate definition at the beginning.")
            qasm_content = p_snippet + "\n" + qasm_content

    if 'cswap' in qasm_content and 'gate cswap' not in qasm_content:
        if re.search(qelib1_pattern, qasm_content):
            qasm_content = re.sub(qelib1_pattern, r'\1' + cswap_snippet, qasm_content)
        else:
            print("Warning -- qelib1.inc not found; appending cswap gate definition at the beginning.")
            qasm_content = cswap_snippet + "\n" + qasm_content

    qasm_content = re.sub(r'\b(u|U)\s*\(', 'u3(', qasm_content)
    qasm_content = re.sub(r'\bcp\s*\(', 'cu1(', qasm_content)

    return qasm_content

def process_qasm_file(path):
    with open(path, 'r') as f:
        raw_qasm = f.read()
    
    clean_qasm = standardize_qasm_gates(raw_qasm)
    
    with open(path, 'w') as f:
        f.write(clean_qasm)

# Extract features from each quantum circuit
eps = 1e-10

# Define weights for different gate types
runtime_weights = {
    'cu1': 15,
    'cx': 10,
    'h': 1,
    'u2': 1.5,
    'measure': 2.,
    'cz': 10.,
    'ry': 1.5,
    'p': 15.,
    'rzz': 15.,
    'u3': 1.5,
    'rccx': 40.,
    'swap': 30.,
    'ccx': 70.,
    'rx': 1.5,
    'u1': 1.5,
    'x': 1.,
    'cswap': 70.,
    'cry': 25.   
}

magic_weights = {
    'cu1': 2.0,
    'u2': 2.0,
    'ry': 2.0,
    'rzz': 3.0,
    'u3': 1.0,
    'rccx': 3.0,
    'ccx': 7.0,
    'rx': 1.0,
    'u1': 1.0,
    'cswap': 7.0,
    'cry': 2.0
}

conditional_gates = ['cu1', 'u2', 'ry', 'rzz', 'u3', 'rccx', 'rx', 'u1', 'cry']

# Functions to calculate gate weights

def calc_runtime_weight(instr):
    gate_name = instr.operation.name
    runtime_weight = runtime_weights.get(gate_name, 0)

    if gate_name in conditional_gates:
        max_runtime_weight = 0

        for param in instr.operation.params:
            indicator = (param / (2*math.pi)) % 1.

            if abs(indicator) < eps:
                weight = runtime_weight / 1.5
            else:
                weight = runtime_weight

            if weight > max_runtime_weight:
                max_runtime_weight = weight

        runtime_weight = max_runtime_weight

    return runtime_weight

def calc_magic_weight(instr):
    gate_name = instr.operation.name
    magic_weight = magic_weights.get(gate_name, 0)

    if gate_name in conditional_gates:
        max_magic_weight = 0

        for param in instr.operation.params:
            indicator = (param / (2*math.pi)) % 1.

            if abs(indicator) < eps:
                weight = 0
            elif abs(indicator - 0.5) < eps:
                weight = 1
            else:
                weight = 3

            if weight > max_magic_weight:
                max_magic_weight = weight

        magic_weight = max_magic_weight

    return magic_weight

def calc_entropy_contribution(qc, instr):

    qb_indices = [qc.find_bit(qb).index for qb in instr.qubits]

    contribution = max(qb_indices) - min(qb_indices)
    return contribution

def calc_domain_size(qc):
    sets = [[i] for i in range(qc.num_qubits)]

    for instr in qc.data:
        if instr.operation.num_qubits > 1:
            qb_indices = [qc.find_bit(qb).index for qb in instr.qubits]

            #print("sets before", sets)
            for set in sets:
                if qb_indices[0] in set:
                    base_set = set
                    break

            #print("base_set:", base_set)

            for idx in qb_indices[1:]:
                for set in sets:
                    if idx in set and set != base_set:
                        base_set.extend(set)
                        sets.remove(set)
                        break

            #print("sets after", sets)

    avg_set_size = 0
    for set in sets:
        avg_set_size += len(set)
    avg_set_size /= qc.num_qubits
    return avg_set_size

def extract_features(qc):
    num_qubits = qc.num_qubits
    depth = qc.depth()

    qc_data = [instr for instr in qc.data if not isinstance(instr.operation, Barrier)]

    gate_counts_by_num_qb = Counter([instr.operation.num_qubits for instr in qc_data])
    mul_qb_gate_count = sum(v for k, v in gate_counts_by_num_qb.items() if k > 1)
    mul_qb_gate_density = mul_qb_gate_count / (num_qubits * depth)

    runtime_weight_sum = 0
    magic_weight_sum = 0
    for instr in qc_data:
        runtime_weight_sum += calc_runtime_weight(instr)
        magic_weight_sum += calc_magic_weight(instr)
    
    weighted_gate_count = runtime_weight_sum
    magic_metric = magic_weight_sum

    entanglement_metric = 0

    for instr in qc_data:
        if instr.operation.num_qubits > 1:
            entanglement_metric += calc_entropy_contribution(qc, instr)

    entanglement_domain_size = calc_domain_size(qc)

    qc_features = {
        'num_qubits': num_qubits,
        'weighted_gate_count': weighted_gate_count,
        'depth': depth,
        'mul_qb_gate_density': mul_qb_gate_density,
        'entanglement_metric': entanglement_metric,
        'magic_metric': magic_metric,
        'entanglement_domain_size': entanglement_domain_size
    }
    return qc_features


def main():
    parser = argparse.ArgumentParser(description="Quantum Fidelity Prediction Pipeline Wrapper")
    
    parser.add_argument("--circuit_dir", type=str, required=True, help="Directory containing QASM files")
    args = parser.parse_args()

    directory_path = Path('circuits')  # Use '.' for the current directory, or specify a path

    for file_path in directory_path.iterdir():
        if file_path.is_file():
            file_name = str(file_path)
            process_qasm_file(file_name)
    print('Refactoring QASM files complete')

    


    qc_arr = []
    qc_names = []

    for file_path in directory_path.iterdir():
        if file_path.is_file():
            file_name = str(file_path)
            qc_names.append(file_name[9:])
            qc_arr.append(qasm.load(file_name))
    print('Loading into Qiskit circuits complete')

    feature_data = []

    for [qc, name] in zip(qc_arr, qc_names):
        features = extract_features(qc)
        features['name'] = name
        feature_data.append(features)

    df_features = pd.DataFrame(feature_data)

    print('Feature extraction complete')

    # Normalization of features

    min_max_columns = ['num_qubits', 'mul_qb_gate_density']
    standard_columns = ['weighted_gate_count', 'depth', 'entanglement_metric', 'magic_metric']

    # Apply log transformation to standardization columns
    df_features[standard_columns] = np.log10(df_features[standard_columns] + 1e-10)

    min_max_scaler = MinMaxScaler()
    standard_scalar = StandardScaler()

    df_scaled = pd.DataFrame()
    df_scaled['name'] = df_features['name']
    df_scaled[standard_columns] = standard_scalar.fit_transform(df_features[standard_columns])
    df_scaled[min_max_columns] = min_max_scaler.fit_transform(df_features[min_max_columns])

    df_scaled.to_csv('qasm_features_scaled.csv', index=False)
    print(df_scaled.head())
    print('Parsing script complete')

if __name__ == "__main__":
    main()