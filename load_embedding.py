import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

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
new_vector = get_qasm_vector("circuits/ae_indep_qiskit_130.qasm")
print(f"Vector Shape: {new_vector.shape}")
print(f"Vector: {new_vector}")