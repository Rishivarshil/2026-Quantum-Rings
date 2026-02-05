import argparse
import json
import os
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import plot_importance

def main():
    parser = argparse.ArgumentParser(description="Predict Runtime")
    
    parser.add_argument("--circuit_dir", type=str, required=True, help="Directory containing QASM files")
    parser.add_argument("--precision", type=str, choices=["single", "double"], required=True, help="Numerical precision")
    parser.add_argument("--backend", type=str, choices=["GPU", "CPU"], required=True, help="Execution backend")
    args = parser.parse_args()
    print("Starting fidelity prediction...")
    for (threshold) in [2, 4, 8, 16, 32, 64, 128, 256]:
        mapping = {
            "precision": {"single": 0, "double": 1},
            "backend": {"GPU": 0, "CPU": 1}
        }

        params_df = pd.DataFrame([{
            "precision": mapping["precision"][args.precision],
            "backend": mapping["backend"][args.backend],
            "normalized_threshold": (1/8) * np.log2(threshold)
        }])

        qasm = pd.read_csv("qasm_features_scaled.csv")
        filtered_row = qasm[qasm['name'] == args.circuit_dir]
        embeddings = pd.read_csv("generated_embeddings.csv")
        if 'Unnamed: 0' in embeddings.columns:
            embeddings = embeddings.drop(columns=['Unnamed: 0'])
        inputs = pd.merge(
            filtered_row, 
            embeddings, 
            left_on=filtered_row.columns[0], 
            right_on=embeddings.columns[0], 
            how='left'
        ).drop(columns=[embeddings.columns[0]])

        for col in params_df.columns:
            inputs[col] = params_df[col].iloc[0]

        cols = list(inputs.columns)
        inputs = inputs[cols[-3:] + cols[:-3]]

        
        X = inputs.dropna()

        loaded_model_sklearn = xgb.XGBRegressor()
        loaded_model_sklearn.load_model("xgb_fidelity_model.json")
        preds = loaded_model_sklearn.predict(X)
        bool = False
        
        for pred in enumerate(preds):
            circuit = args.circuit_dir
            print(f"Circuit: {circuit}, Threshold: {threshold}, Predicted Fidelity: {pred[1]}")
            circuit_name = os.path.basename(circuit)
            if pred[1] >= 0.79:  
                json.dump(
                    {
                        "circuit": circuit_name,
                        "threshold": threshold,
                        "predicted_fidelity": float(pred[1]),
                    },
                    open(f"fidelity_prediction_{circuit_name}.json", "w")
                )                
                bool = True
                break

        if bool:
            break
        else:
            continue
        


if __name__ == "__main__":
    main()