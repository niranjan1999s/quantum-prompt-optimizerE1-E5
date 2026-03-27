from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class HighCapacityQuantumAgent(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 20, embed_dim: int = 768, entanglement_mode: str = "full") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement_mode = entanglement_mode

        self.q_params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params):
            if self.entanglement_mode == "full":
                qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            else:
                # Separable or no_entangle: Just the rotations from StronglyEntanglingLayers without any entangling CNOTs
                for layer in range(self.n_layers):
                    for qubit in range(self.n_qubits):
                        qml.Rot(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], wires=qubit)
                        
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = circuit
        self.projector = nn.Linear(n_qubits, embed_dim)

    def forward(self, x=None) -> torch.Tensor:
        q_out = self.qlayer(self.q_params)
        q_out = torch.stack(q_out).float()
        return self.projector(q_out)

