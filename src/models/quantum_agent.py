from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class HighCapacityQuantumAgent(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 20, embed_dim: int = 768,
                 entanglement_mode: str = "full", noise_rate: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement_mode = entanglement_mode
        self.noise_rate = noise_rate

        self.q_params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Use density-matrix simulator when noise is present
        if noise_rate > 0.0:
            dev = qml.device("default.mixed", wires=n_qubits)
        else:
            dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(params):
            if self.noise_rate > 0.0:
                # Manually decompose StronglyEntanglingLayers + inject noise after each layer
                for layer_idx in range(self.n_layers):
                    # Single-qubit rotations
                    for qubit in range(self.n_qubits):
                        qml.Rot(params[layer_idx, qubit, 0],
                                params[layer_idx, qubit, 1],
                                params[layer_idx, qubit, 2], wires=qubit)
                    # CNOT ring (same pattern as StronglyEntanglingLayers)
                    if self.entanglement_mode == "full":
                        for qubit in range(self.n_qubits):
                            target = (qubit + 1 + layer_idx) % self.n_qubits
                            if target != qubit:
                                qml.CNOT(wires=[qubit, target])
                    # Depolarizing noise on every qubit after each layer
                    for qubit in range(self.n_qubits):
                        qml.DepolarizingChannel(self.noise_rate, wires=qubit)
            else:
                if self.entanglement_mode == "full":
                    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
                else:
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

