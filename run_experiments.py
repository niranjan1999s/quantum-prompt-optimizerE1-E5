import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

seeds = list(range(10))  # 0 to 9
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E4: 8-Qubit Scaled Sweep (10 seeds on SST-2)")
print(f"Config: {n_qubits} qubits x {n_layers} layers = {n_qubits * n_layers * 3} VQC params")
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
        "--checkpoint-every", "0",
        "--print-every", "1"
    ]
    subprocess.run(cmd, check=True)

print("\nE4 experiments completed successfully.")
