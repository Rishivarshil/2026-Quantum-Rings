import json
import pandas as pd

def get_expected_runtime_sec(result):
    forward = result.get("forward")
    if forward and forward.get("run_wall_s") is not None:
        return forward["run_wall_s"]

    est = result.get("forward_timing_estimates") or {}
    setup = est.get("estimated_setup_s")
    per_shot = est.get("estimated_per_shot_s")
    if setup is not None and per_shot is not None:
        return setup + 10_000 * per_shot

    return None


def extract_to_dataframe(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []

    for result in data.get("results", []):
        file_name = result.get("file")
        precision = result.get("precision")
        backend = result.get("backend")

        # compute forward runtime once per result (fallback only)
        forward_runtime = get_expected_runtime_sec(result)

        for sweep in result.get("threshold_sweep", []):
            sweep_runtime = sweep.get("run_wall_s")

            records.append({
                "circuit": file_name,
                "precision": precision,
                "backend": backend,
                "threshold": sweep.get("threshold"),
                "fidelity": sweep.get("sdk_get_fidelity"),
                # 🔹 ONLY replace when sweep runtime is missing
                "expected_runtime_sec": (
                    sweep_runtime if sweep_runtime is not None else forward_runtime
                ),
            })

    return pd.DataFrame(records)


df = extract_to_dataframe("data/hackathon_public.json")

pd.set_option("display.max_rows", None)

df.to_csv("extracted_public_data.csv", index=False)