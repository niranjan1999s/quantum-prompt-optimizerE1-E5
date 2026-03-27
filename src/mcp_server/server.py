from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from training.trainer import TrainConfig, train_sequence_optimization

app = FastAPI(title="quantum-prompt-optimizer")


class TrainRequest(BaseModel):
    hard_prompt_text: str = TrainConfig.hard_prompt_text
    target_word: str = TrainConfig.target_word
    epochs: int = TrainConfig.epochs
    lr: float = TrainConfig.lr
    n_qubits: int = TrainConfig.n_qubits
    n_layers: int = TrainConfig.n_layers
    embed_dim: int = TrainConfig.embed_dim
    seed: int = TrainConfig.seed
    print_every: int = 0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(req: TrainRequest) -> dict:
    cfg = TrainConfig(
        hard_prompt_text=req.hard_prompt_text,
        target_word=req.target_word,
        epochs=req.epochs,
        lr=req.lr,
        n_qubits=req.n_qubits,
        n_layers=req.n_layers,
        embed_dim=req.embed_dim,
        seed=req.seed,
        print_every=req.print_every,
        plot=False,
    )
    result = train_sequence_optimization(cfg)
    return {
        "epochs": cfg.epochs,
        "final_quantum_loss": result["quantum_loss_history"][-1],
        "final_classical_loss": result["classical_loss_history"][-1],
    }

