from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np

from models import ClassicalGoliathAgent, ClassicalDavidAgent, HighCapacityQuantumAgent

@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    model_name: str = "gpt2"
    epochs: int = 10
    lr: float = 0.01
    n_qubits: int = 8
    n_layers: int = 30
    embed_dim: int = 768
    seed: int = 0
    batch_size: int = 16
    print_every: int = 1
    checkpoint_every: int = 10
    entanglement_mode: str = "full"
    plot: bool = False

def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

def train_classification(cfg: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    llm = GPT2LMHeadModel.from_pretrained(cfg.model_name).to(device)
    _freeze(llm)

    quantum_agent = HighCapacityQuantumAgent(
        n_qubits=cfg.n_qubits, n_layers=cfg.n_layers, embed_dim=cfg.embed_dim,
        entanglement_mode=cfg.entanglement_mode
    ).to(device)
    
    q_quantum_only = cfg.n_layers * cfg.n_qubits * 3
    q_total = sum(p.numel() for p in quantum_agent.parameters() if p.requires_grad)
    
    david_agent = ClassicalDavidAgent(
        n_inputs=cfg.n_qubits, embed_dim=cfg.embed_dim, exact_params=q_total
    ).to(device)
    # Goliath scales its hidden layers up with n_inputs so it always remains the gigantic baseline
    classical_agent = ClassicalGoliathAgent(n_inputs=cfg.n_qubits, embed_dim=cfg.embed_dim).to(device)

    d_total = sum(p.numel() for p in david_agent.parameters() if p.requires_grad)
    c_total = sum(p.numel() for p in classical_agent.parameters() if p.requires_grad)

    exp_root = Path("experiments") / cfg.model_name / cfg.dataset_config
    exp_root.mkdir(parents=True, exist_ok=True)
    run_dir = exp_root / f"seed{cfg.seed}_ep{cfg.epochs}_{cfg.entanglement_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Dataset loading
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    train_data = dataset["train"].select(range(200))   # Sliced for E4 speed (GTX 1650)
    
    val_split_name = "validation" if "validation" in dataset else "test"
    val_data = dataset[val_split_name].select(range(max(1, min(500, len(dataset[val_split_name])))))
    
    def collate_fn(batch):
        if cfg.dataset_config == "sst2":
            texts = [f"Review: {item['sentence']}\nSentiment:" for item in batch]
        elif cfg.dataset_config == "rte":
            texts = [f"Premise: {item['sentence1']}\nHypothesis: {item['sentence2']}\nEntailment:" for item in batch]
        elif cfg.dataset_config == "boolq":
            texts = [f"Passage: {item['passage']}\nQuestion: {item['question']}\nAnswer:" for item in batch]
        elif cfg.dataset_name == "ag_news":
            texts = [f"Article: {item['text']}\nCategory:" for item in batch]
        else:
            raise ValueError(f"Unsupported dataset config: {cfg.dataset_config}")
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        encodings = tokenizer(texts, padding=True, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    if cfg.dataset_config == "sst2":
        # Verbalizer Map: SST-2 labels are 0: Negative, 1: Positive
        token_neg = tokenizer.encode(" Negative")[0]
        token_pos = tokenizer.encode(" Positive")[0]
        verbalizer = torch.tensor([token_neg, token_pos]).to(device)
    elif cfg.dataset_config == "rte":
        # Verbalizer Map: RTE labels are 0: entailment (Yes), 1: not_entailment (No)
        token_yes = tokenizer.encode(" Yes")[0]
        token_no = tokenizer.encode(" No")[0]
        verbalizer = torch.tensor([token_yes, token_no]).to(device)
    elif cfg.dataset_config == "boolq":
        # Verbalizer Map: BoolQ labels are 0: False (No), 1: True (Yes)
        token_yes = tokenizer.encode(" Yes")[0]
        token_no = tokenizer.encode(" No")[0]
        verbalizer = torch.tensor([token_no, token_yes]).to(device)
    elif cfg.dataset_name == "ag_news":
        # Verbalizer Map: AG News labels are 0: World, 1: Sports, 2: Business, 3: Sci/Tech
        token_world = tokenizer.encode(" World")[0]
        token_sports = tokenizer.encode(" Sports")[0]
        token_business = tokenizer.encode(" Business")[0]
        token_tech = tokenizer.encode(" Tech")[0]
        verbalizer = torch.tensor([token_world, token_sports, token_business, token_tech]).to(device)
    else:
        raise ValueError(f"Unsupported dataset config: {cfg.dataset_config}")

    dummy_input = torch.zeros(cfg.n_qubits).to(device)

    opt_quantum = optim.Adam(quantum_agent.parameters(), lr=cfg.lr)
    opt_david = optim.Adam(david_agent.parameters(), lr=cfg.lr)
    opt_classical = optim.Adam(classical_agent.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    quantum_loss_history: list[float] = []
    david_loss_history: list[float] = []
    classical_loss_history: list[float] = []
    
    quantum_acc_history: list[float] = []
    david_acc_history: list[float] = []
    classical_acc_history: list[float] = []

    for epoch in range(cfg.epochs):
        quantum_agent.train()
        david_agent.train()
        classical_agent.train()
        
        q_epoch_loss, d_epoch_loss, c_epoch_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            target_ids = verbalizer[labels] 
            
            # Since pad_side='left', the last token is always at sequence length!
            batch_sz, seq_len = input_ids.shape
            # Get hard embeddings
            hard_embeddings = llm.transformer.wte(input_ids)
            
            # --- Quantum Agent ---
            opt_quantum.zero_grad(set_to_none=True)
            q_prompt = quantum_agent() # shape: (768,)
            q_soft_prompt = q_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
            q_input_embeds = torch.cat([q_soft_prompt, hard_embeddings], dim=1)
            q_outputs = llm(inputs_embeds=q_input_embeds)
            q_logits = q_outputs.logits[:, -1, :] # Predict next token after "Sentiment:"
            q_loss = loss_fn(q_logits, target_ids)
            q_loss.backward()
            opt_quantum.step()
            q_epoch_loss += q_loss.item()
            
            # --- David Agent ---
            opt_david.zero_grad(set_to_none=True)
            d_prompt = david_agent(dummy_input)
            d_soft_prompt = d_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
            d_input_embeds = torch.cat([d_soft_prompt, hard_embeddings], dim=1)
            d_outputs = llm(inputs_embeds=d_input_embeds)
            d_logits = d_outputs.logits[:, -1, :]
            d_loss = loss_fn(d_logits, target_ids)
            d_loss.backward()
            opt_david.step()
            d_epoch_loss += d_loss.item()
            
            # --- Classical Goliath Agent ---
            opt_classical.zero_grad(set_to_none=True)
            c_prompt = classical_agent(dummy_input)
            c_soft_prompt = c_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
            c_input_embeds = torch.cat([c_soft_prompt, hard_embeddings], dim=1)
            c_outputs = llm(inputs_embeds=c_input_embeds)
            c_logits = c_outputs.logits[:, -1, :]
            c_loss = loss_fn(c_logits, target_ids)
            c_loss.backward()
            opt_classical.step()
            c_epoch_loss += c_loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | QL: {q_loss.item():.3f} DL: {d_loss.item():.3f} CL: {c_loss.item():.3f}")
                import sys; sys.stdout.flush()
                
        # Epoch averages
        quantum_loss_history.append(q_epoch_loss / len(train_loader))
        david_loss_history.append(d_epoch_loss / len(train_loader))
        classical_loss_history.append(c_epoch_loss / len(train_loader))
        
        # Validation Loop
        quantum_agent.eval()
        david_agent.eval()
        classical_agent.eval()
        
        q_correct, d_correct, c_correct, total = 0, 0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                target_ids = verbalizer[labels]
                batch_sz = input_ids.shape[0]
                hard_embeddings = llm.transformer.wte(input_ids)
                
                # Quantum evaluates
                q_prompt = quantum_agent()
                q_soft_prompt = q_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
                q_logits = llm(inputs_embeds=torch.cat([q_soft_prompt, hard_embeddings], dim=1)).logits[:, -1, :]
                
                # David evaluates
                d_prompt = david_agent(dummy_input)
                d_soft_prompt = d_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
                d_logits = llm(inputs_embeds=torch.cat([d_soft_prompt, hard_embeddings], dim=1)).logits[:, -1, :]
                
                # Classical evaluates
                c_prompt = classical_agent(dummy_input)
                c_soft_prompt = c_prompt.unsqueeze(0).unsqueeze(0).expand(batch_sz, 1, -1)
                c_logits = llm(inputs_embeds=torch.cat([c_soft_prompt, hard_embeddings], dim=1)).logits[:, -1, :]
                
                # Accuracy calculation by comparing the verbalizer logits
                q_preds = torch.argmax(q_logits[:, verbalizer], dim=-1)
                d_preds = torch.argmax(d_logits[:, verbalizer], dim=-1)
                c_preds = torch.argmax(c_logits[:, verbalizer], dim=-1)
                
                q_correct += (q_preds == labels).sum().item()
                d_correct += (d_preds == labels).sum().item()
                c_correct += (c_preds == labels).sum().item()
                total += batch_sz
                
        q_acc = q_correct / total
        d_acc = d_correct / total
        c_acc = c_correct / total
        
        quantum_acc_history.append(q_acc)
        david_acc_history.append(d_acc)
        classical_acc_history.append(c_acc)

        if cfg.checkpoint_every > 0 and (epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs - 1):
            ckpt_path = run_dir / f"ckpt_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "config": asdict(cfg),
                "quantum_state": quantum_agent.state_dict(),
                "david_state": david_agent.state_dict(),
                "classical_state": classical_agent.state_dict(),
                "opt_quantum": opt_quantum.state_dict(),
                "opt_david": opt_david.state_dict(),
                "opt_classical": opt_classical.state_dict(),
                "quantum_loss_history": quantum_loss_history,
                "david_loss_history": david_loss_history,
                "classical_loss_history": classical_loss_history,
                "quantum_acc_history": quantum_acc_history,
                "david_acc_history": david_acc_history,
                "classical_acc_history": classical_acc_history,
            }, ckpt_path)

        if cfg.print_every > 0 and epoch % cfg.print_every == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Q-Acc: {q_acc:.3f} | "
                f"D-Acc: {d_acc:.3f} | "
                f"C-Acc: {c_acc:.3f}"
            )

    result: dict[str, Any] = {
        "config": asdict(cfg),
        "quantum_loss_history": quantum_loss_history,
        "david_loss_history": david_loss_history,
        "classical_loss_history": classical_loss_history,
        "quantum_acc_history": quantum_acc_history,
        "david_acc_history": david_acc_history,
        "classical_acc_history": classical_acc_history,
        "q_quantum_only": q_quantum_only,
        "q_total_params": q_total,
        "d_total_params": d_total,
        "c_total_params": c_total,
        "run_dir": str(run_dir),
    }

    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quantum vs Classical soft-prompt SST2 optimizer.")
    p.add_argument("--dataset-name", default=TrainConfig.dataset_name)
    p.add_argument("--dataset-config", default=TrainConfig.dataset_config)
    p.add_argument("--model-name", default=TrainConfig.model_name)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--n-qubits", type=int, default=TrainConfig.n_qubits)
    p.add_argument("--n-layers", type=int, default=TrainConfig.n_layers)
    p.add_argument("--embed-dim", type=int, default=TrainConfig.embed_dim)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--print-every", type=int, default=TrainConfig.print_every)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    p.add_argument("--entanglement-mode", default=TrainConfig.entanglement_mode,
                    choices=["full", "no_entangle", "separable"])
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        embed_dim=args.embed_dim,
        seed=args.seed,
        batch_size=args.batch_size,
        print_every=args.print_every,
        checkpoint_every=args.checkpoint_every,
        entanglement_mode=args.entanglement_mode,
        plot=args.plot,
    )
    train_classification(cfg)

if __name__ == "__main__":
    main()
