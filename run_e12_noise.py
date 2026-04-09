"""
E12: Noise Model Simulation
Tests VQC robustness under realistic quantum hardware conditions by injecting
depolarizing noise at varying error rates during SST-2 training.
10 seeds per noise level, 8 qubits x 30 layers.
"""
import subprocess

PYTHON = r".\.venv\Scripts\python.exe"

noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05]
seeds = list(range(10))
model = "gpt2"
epochs = 5
n_qubits = 8
n_layers = 30

print("==========================================")
print("Starting E12: Noise Model Simulation")
print(f"Config: {n_qubits} qubits x {n_layers} layers")
print(f"Noise levels: {noise_levels}")
print(f"Seeds: {len(seeds)} per level = {len(noise_levels) * len(seeds)} total runs")
print("==========================================")

for noise in noise_levels:
    print(f"\n========== Noise Rate: {noise} ==========")
    for seed in seeds:
        print(f"\n--- Noise {noise} | Seed: {seed} ---")
        cmd = [
            PYTHON, "-u", "train.py",
            "--dataset-name", "glue",
            "--dataset-config", "sst2",
            "--model-name", model,
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--n-qubits", str(n_qubits),
            "--n-layers", str(n_layers),
            "--batch-size", "8",
            "--checkpoint-every", "0",
            "--print-every", "1",
            "--noise-rate", str(noise)
        ]
        subprocess.run(cmd, check=True)

print("\nE12 noise simulation completed successfully.")
