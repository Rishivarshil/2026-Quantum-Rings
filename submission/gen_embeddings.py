import argparse
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
from pathlib import Path
import pandas as pd

# 1. Load the trained model
model = Doc2Vec.load("qasm_doc2vec.model")

def get_qasm_vector(file_path):
    # 2. Read the new QASM file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3. Preprocess exactly like training (CRITICAL: min_len=1)
    # This turns "h q[0];" into ['h', 'q', '0']
    tokens = simple_preprocess(content, min_len=1)

    # 4. Infer the vector
    # 'steps' is how many times it re-runs the inference to fine-tune the vector
    vector = model.infer_vector(tokens, epochs=50)
    
    return vector

# Example Usage
def main():
    parser = argparse.ArgumentParser(description="Quantum Fidelity Prediction Pipeline Wrapper")
    
    parser.add_argument("--circuit_dir", type=str, required=True, help="Directory containing QASM files")
    args = parser.parse_args()


    name_list = [args.circuit_dir]
    emb_list = [get_qasm_vector(args.circuit_dir)]



    column_names = [str(i) for i in range(50)]

    df_names = pd.DataFrame(name_list, columns =['name'])
    df_embs = pd.DataFrame(emb_list, columns=column_names)

    df = pd.concat([df_names, df_embs], axis=1)

    df.to_csv('generated_embeddings.csv')
    print('Successful generation!')

if __name__ == "__main__":
    main()