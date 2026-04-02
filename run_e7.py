"""
E7: Question Answering Experiment (BoolQ)
Runs 10 seeds on the BoolQ dataset to evaluate reading comprehension generalization.
"""
import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

seeds = list(range(10))
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E7: BoolQ Question Answering (10 seeds on SuperGLUE BoolQ)")
print(f"Config: {n_qubits} qubits x {n_layers} layers")
print("==========================================")

for seed in seeds:
    print(f"\n--- Running Seed: {seed} ---")
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset-name", "super_glue",
        "--dataset-config", "boolq",
        "--model-name", model,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--n-qubits", str(n_qubits),
        "--n-layers", str(n_layers),
        "--batch-size", "2",
        "--checkpoint-every", "0",
        "--print-every", "1"
    ]
    subprocess.run(cmd, check=True)

print("\nE7 experiments completed successfully.")
