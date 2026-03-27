## quantum-prompt-optimizer

This project is a structured version of the notebook in `notebooks/source.ipynb`, split into:

- `src/models/quantum_agent.py`: `HighCapacityQuantumAgent` (PennyLane + PyTorch)
- `src/models/classical_agent.py`: `ClassicalGoliathAgent` (PyTorch baseline)
- `src/training/trainer.py`: training loop that races both agents on a GPT-2 soft-prompt task

### Quick start (Windows)

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py --epochs 50 --plot
```

Change the target word / prompt:

```bash
python train.py --hard-prompt-text "The quantum key distribution protocol used is" --target-word " thermo-nuclear-astrophysics"
```

### Optional: run the API server

```bash
uvicorn mcp_server.server:app --app-dir src --reload
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

