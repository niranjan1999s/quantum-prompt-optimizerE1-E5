"""
E6: Natural Language Inference Experiment (RTE)
Runs 10 seeds on the RTE dataset to evaluate generalization.
"""
import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

seeds = list(range(10))
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E6: Recognizing Textual Entailment (10 seeds on RTE)")
print(f"Config: {n_qubits} qubits x {n_layers} layers")
print("==========================================")

for seed in seeds:
    print(f"\n--- Running Seed: {seed} ---")
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset-config", "rte",
        "--model-name", model,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--n-qubits", str(n_qubits),
        "--n-layers", str(n_layers),
        "--batch-size", "8",
        "--checkpoint-every", "0",
        "--print-every", "1"
    ]
    subprocess.run(cmd, check=True)

print("\nE6 experiments completed successfully.")
