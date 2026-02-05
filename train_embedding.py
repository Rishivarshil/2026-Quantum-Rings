import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

DATA_DIRS = [
    #"./mnisq_datasets/base_train_orig_Kuzushiji-MNIST_f80/qasm",
    #"./mnisq_datasets/base_train_orig_Kuzushiji-MNIST_f90/qasm",
    "./mnisq_datasets/base_train_orig_Kuzushiji-MNIST_f95/qasm",
    #"./mnisq_datasets/base_train_orig_Fashion-MNIST_f80/qasm",
    #"./mnisq_datasets/base_train_orig_Fashion-MNIST_f90/qasm",
    #"./mnisq_datasets/base_train_orig_Fashion-MNIST_f95/qasm",
    #"./mnisq_datasets/base_train_orig_mnist_784_f80/qasm",
    #"./mnisq_datasets/base_train_orig_mnist_784_f90/qasm"
    #"./mnisq_datasets/base_train_orig_mnist_784_f95/qasm"
    # Add as many paths as you need
]

MODEL_PATH = "qasm_doc2vec.model"
VECTOR_SIZE = 50             
EPOCHS = 40                  

def read_corpus(directories):
    """
    Reads QASM files from a LIST of directories and yields TaggedDocument objects.
    """
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping.")
            continue
        
        # Get the folder name to make tags unique (e.g., 'dataset1_file.qasm')
        dir_name = os.path.basename(os.path.normpath(directory))

        for filename in os.listdir(directory):
            # Check if it's a file (skips sub-directories)
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read()
                    except UnicodeDecodeError:
                        print(f"Skipping {filename} due to encoding error.")
                        continue
                
                # TOKENIZATION:
                tokens = simple_preprocess(content, min_len=1)
                
                # UNIQUE TAGGING:
                # We combine directory name + filename to avoid collisions 
                # if two folders have a file named "circuit_0.qasm"
                unique_tag = f"{dir_name}/{filename}"
                
                yield TaggedDocument(words=tokens, tags=[unique_tag])
def train_qasm_model():
    print("1. Loading and preprocessing data from multiple directories...")
    # Pass the list DATA_DIRS directly
    train_corpus = list(read_corpus(DATA_DIRS))
    
    if not train_corpus:
        print("No files found in any of the provided directories!")
        return

    print(f"   Found {len(train_corpus)} scripts total.")

    print("2. Initializing Doc2Vec model...")
    model = Doc2Vec(vector_size=VECTOR_SIZE, 
                    min_count=1,      
                    epochs=EPOCHS,
                    dm=1,             
                    window=5,         
                    workers=4)

    print("3. Building Vocabulary...")
    model.build_vocab(train_corpus)

    print("4. Training...")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print("5. Saving model...")
    model.save(MODEL_PATH)
    print(f"   Model saved to '{MODEL_PATH}'")
    
    return model

if __name__ == "__main__":
    model = train_qasm_model()