"""
E3: Entanglement Ablation Experiment
Runs 10 seeds with entanglement_mode="no_entangle" to compare against E4's "full" results.
Same 8-qubit, 30-layer config — only the CNOT gates are stripped.
"""
import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

seeds = list(range(10))
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E3: Entanglement Ablation (10 seeds on SST-2)")
print(f"Config: {n_qubits} qubits x {n_layers} layers, mode=no_entangle")
print("==========================================")

for seed in seeds:
    print(f"\n--- Running Seed: {seed} ---")
    cmd = [
        PYTHON, "-u", "train.py",
        "--model-name", model,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--n-qubits", str(n_qubits),
        "--n-layers", str(n_layers),
        "--entanglement-mode", "no_entangle",
        "--checkpoint-every", "0",
        "--print-every", "1"
    ]
    subprocess.run(cmd, check=True)

print("\nE3 experiments completed successfully.")
