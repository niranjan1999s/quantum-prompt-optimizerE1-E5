"""
E8: Topic Classification Experiment (AG News)
Runs 10 seeds on the AG News dataset to evaluate 4-class categorization.
"""
import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

seeds = list(range(10))
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E8: AG News Topic Classification (10 seeds)")
print(f"Config: {n_qubits} qubits x {n_layers} layers")
print("==========================================")

for seed in seeds:
    print(f"\n--- Running Seed: {seed} ---")
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset-name", "ag_news",
        "--dataset-config", "default",
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

print("\nE8 experiments completed successfully.")
