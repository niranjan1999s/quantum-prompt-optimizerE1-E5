"""
E9: Cross-Task Transfer Matrix
Takes a trained quantum state from Task A and evaluates it zero-shot on Task B.
"""
import subprocess
from pathlib import Path

PYTHON = r".\.venv\Scripts\python.exe"

tasks = [
    {"name": "glue", "config": "sst2", "batch": "8"},
    {"name": "glue", "config": "rte", "batch": "8"},
    {"name": "super_glue", "config": "boolq", "batch": "2"},
    {"name": "ag_news", "config": "default", "batch": "8"}
]

print("==========================================")
print("E9: Stage 1 - Rebuilding Seed 0 Checkpoints")
print("==========================================")

# Since `--checkpoint-every 0` was set across E6-E8 to save SSD space, we have to
# quickly re-train just Seed 0 over 5 epochs for all 4 datasets to generate the `.pt` models!

base_dir = Path("experiments/gpt2")

for t in tasks:
    tgt_folder = base_dir / t["config"] / "seed0_ep5_full"
    # Check if a .pt file exists inside
    if tgt_folder.exists() and list(tgt_folder.glob("*.pt")):
        print(f"[*] Checkpoint already exists for {t['config']}! Skipping training.")
        continue
        
    print(f"\n[!] Missing Checkpoint for {t['config']}. Rebuilding Seed 0...")
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset-name", t["name"],
        "--dataset-config", t["config"],
        "--epochs", "5",
        "--seed", "0",
        "--n-qubits", "8",
        "--n-layers", "30",
        "--batch-size", t["batch"],
        "--checkpoint-every", "5",  # FORCE SAVE THE PT AT THE END
        "--print-every", "1"
    ]
    subprocess.run(cmd, check=True)

print("\n==========================================")
print("E9: Stage 2 - Executing Cross-Task Transfer Matrix")
print("==========================================")

def get_best_seed0_ckpt(config_name):
    config_dir = base_dir / config_name
    seed_dirs = list(config_dir.glob("seed0_*"))
    if not seed_dirs: return None
    target_dir = seed_dirs[0]
    ckpts = list(target_dir.glob("ckpt_epoch_*.pt"))
    if not ckpts: return None
    ckpts.sort()
    return ckpts[-1]

source_ckpts = {}
for t in tasks:
    ckpt = get_best_seed0_ckpt(t["config"])
    if ckpt: source_ckpts[t["config"]] = ckpt

for src_config, ckpt_path in source_ckpts.items():
    print(f"\n>>> Source Model: {src_config.upper()} <<<")
    for tgt in tasks:
        print(f"  -> Transferring zero-shot to: {tgt['config']}")
        cmd = [
            PYTHON, "-u", "train.py",
            "--dataset-name", tgt["name"],
            "--dataset-config", tgt["config"],
            "--eval-only",
            "--load-ckpt", str(ckpt_path),
            "--batch-size", "8", 
            "--print-every", "1" 
        ]
        subprocess.run(cmd, check=True)

print("\nE9 Transfer Matrix loops completed!")
