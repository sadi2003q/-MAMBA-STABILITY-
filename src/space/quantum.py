"""
A simulated quantum circuit that can replace Chain-Mamba's gate layer (x_proj), plus hardware defects.

----------------------------------------------------------------------------------------------------------
Where the circuit goes
----------------------------------------------------------------------------------------------------------
In every Chain-Mamba layer with memory, one linear layer `x_proj` turns the memory-branch signal xc into
three gates: the step size dt, B (what enters the memory) and C (what is read out). See _selective() in
src/models/mamba_chain.py. The quantum version replaces ONLY that linear layer:

    xc [.., E]  ->  squeeze to 6 numbers  ->  6 angles in (-pi, pi)  ->  6-qubit circuit  ->  12 outputs
                ->  expand to dt_rank + 2 x d_state numbers  (the same size as before)

Everything else in Chain-Mamba stays exactly as it is, including the decay margin.

----------------------------------------------------------------------------------------------------------
The circuit (simulated exactly on the computer)
----------------------------------------------------------------------------------------------------------
    start with every qubit in state |0>
    repeat `layers` times:  turn each qubit by an RY rotation using its input angle, then a ring of CNOT gates
    finally one more trainable RY rotation on each qubit
    measure Z on each qubit (6 numbers) and Z x Z on each neighbouring pair (6 numbers)
All 12 outputs are between -1 and +1. Only RY and CNOT gates are used, so every amplitude stays a real number.
Qubit 0 is the most significant bit of the basis-state index.

----------------------------------------------------------------------------------------------------------
Hardware defects (all zero = perfect hardware), applied in the order they happen physically
----------------------------------------------------------------------------------------------------------
    1. gate_error g:     imperfect operations shrink every answer toward zero:  Z x (1-g),  ZZ x (1-g)^2
    2. readout a, b:     a 0 is misread as 1 with probability a, a 1 as 0 with probability b. Exact effect:
                         Z -> s Z + c,   ZZ -> s^2 ZZ + s c (Z_i + Z_j) + c^2,   s = 1 - a - b,  c = b - a
    3. shots N:          each answer is an average of N measurements: random scatter, variance (1 - value^2) / N
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn


class QuantumCircuitSimulator(nn.Module):
    def __init__(self, number_of_qubits: int, number_of_layers: int):
        super().__init__()

        # Step 1: sizes
        self.number_of_qubits = number_of_qubits
        self.number_of_layers = number_of_layers
        self.number_of_states = 2 ** number_of_qubits

        # Step 2: hardware defects (set from outside with set_hardware(); all zero = perfect hardware)
        self.shots = None
        self.gate_error = 0.0
        self.readout_a = 0.0
        self.readout_b = 0.0

        # Step 3: trainable circuit numbers
        self.input_scaling = nn.Parameter(torch.ones(number_of_layers, number_of_qubits))
        self.layer_angles = nn.Parameter(torch.empty(number_of_layers, number_of_qubits).uniform_(-math.pi, math.pi))
        self.final_angles = nn.Parameter(torch.empty(number_of_qubits).uniform_(-math.pi, math.pi))

        # Step 4: the value of every measured quantity in every basis state: Z (6) and Z x Z (6)
        basis = torch.arange(self.number_of_states)
        bit_columns = []
        for qubit in range(number_of_qubits):
            bit_columns.append((basis >> (number_of_qubits - 1 - qubit)) & 1)
        bits = torch.stack(bit_columns, dim=1)                              # [states, qubits]
        z_values = 1.0 - 2.0 * bits.float()                                 # bit 0 -> +1, bit 1 -> -1
        zz_columns = []
        for qubit in range(number_of_qubits):
            neighbour = (qubit + 1) % number_of_qubits
            zz_columns.append(z_values[:, qubit] * z_values[:, neighbour])
        zz_values = torch.stack(zz_columns, dim=1)
        self.register_buffer("observables", torch.cat([z_values, zz_values], dim=1))   # [states, 12]

        # Step 5: the whole ring of CNOT gates written as ONE reordering of the basis states
        combined = torch.arange(self.number_of_states)
        for control in range(number_of_qubits):
            target = (control + 1) % number_of_qubits
            control_bit = (basis >> (number_of_qubits - 1 - control)) & 1
            flipped = basis ^ (1 << (number_of_qubits - 1 - target))
            one_cnot = torch.where(control_bit == 1, flipped, basis)
            combined = combined[one_cnot]
        self.register_buffer("cnot_ring_permutation", combined)

    def output_size(self) -> int:
        return 2 * self.number_of_qubits

    # ------------------------------------------------------------------------------------------------------
    def apply_ry(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """Turn one qubit by an RY rotation. state [N, states], angle [N]."""
        samples = state.shape[0]
        left = 2 ** qubit
        right = 2 ** (self.number_of_qubits - qubit - 1)
        reshaped = state.reshape(samples, left, 2, right)
        amplitude_0 = reshaped[:, :, 0, :]
        amplitude_1 = reshaped[:, :, 1, :]
        cosine = torch.cos(angle / 2).reshape(samples, 1, 1)
        sine = torch.sin(angle / 2).reshape(samples, 1, 1)
        new_0 = cosine * amplitude_0 - sine * amplitude_1
        new_1 = sine * amplitude_0 + cosine * amplitude_1
        return torch.stack([new_0, new_1], dim=2).reshape(samples, self.number_of_states)

    def exact_outputs(self, input_angles: torch.Tensor) -> torch.Tensor:
        """Perfect-hardware outputs. input_angles [N, qubits] -> [N, 12]."""
        samples = input_angles.shape[0]

        # Step 1: all qubits start in |0>
        state = torch.zeros(samples, self.number_of_states, device=input_angles.device, dtype=input_angles.dtype)
        state[:, 0] = 1.0

        # Step 2: the repeated layers: rotations using the input, then the CNOT ring
        for layer in range(self.number_of_layers):
            for qubit in range(self.number_of_qubits):
                angle = self.input_scaling[layer, qubit] * input_angles[:, qubit] + self.layer_angles[layer, qubit]
                state = self.apply_ry(state, qubit, angle)
            state = state[:, self.cnot_ring_permutation]

        # Step 3: final trainable rotation on each qubit
        for qubit in range(self.number_of_qubits):
            angle = self.final_angles[qubit].expand(samples)
            state = self.apply_ry(state, qubit, angle)

        # Step 4: measured averages: probability of each basis state x value of each quantity
        probabilities = state * state
        return probabilities @ self.observables

    def forward(self, input_angles: torch.Tensor) -> torch.Tensor:
        # Step 1: perfect-hardware answers
        measured = self.exact_outputs(input_angles)
        n = self.number_of_qubits
        z = measured[:, :n]
        zz = measured[:, n:]

        # Step 2: gate errors shrink every answer toward zero
        if self.gate_error > 0:
            z = z * (1.0 - self.gate_error)
            zz = zz * (1.0 - self.gate_error) ** 2

        # Step 3: readout errors (Z x Z uses the Z values from BEFORE the readout step)
        if self.readout_a > 0 or self.readout_b > 0:
            s = 1.0 - self.readout_a - self.readout_b
            c = self.readout_b - self.readout_a
            z_neighbour = torch.roll(z, shifts=-1, dims=1)
            zz = s * s * zz + s * c * (z + z_neighbour) + c * c
            z = s * z + c

        measured = torch.cat([z, zz], dim=1)

        # Step 4: limited number of measurements -> random scatter
        if self.shots is not None:
            variance = torch.clamp(1.0 - measured * measured, min=0.0) / self.shots
            measured = measured + torch.randn_like(measured) * torch.sqrt(variance)
            measured = torch.clamp(measured, -1.0, 1.0)
        return measured


class QuantumSelection(nn.Module):
    """Drop-in replacement for Chain-Mamba's x_proj: same input size E, same output size."""

    def __init__(self, input_size: int, output_size: int, number_of_qubits: int, number_of_layers: int):
        super().__init__()
        self.number_of_qubits = number_of_qubits
        self.squeeze = nn.Linear(input_size, number_of_qubits)
        self.circuit = QuantumCircuitSimulator(number_of_qubits, number_of_layers)
        self.expand = nn.Linear(self.circuit.output_size(), output_size)

    def forward(self, xc: torch.Tensor) -> torch.Tensor:
        # Step 1: remember the leading shape ([B, T] for a sequence, [B] for one step)
        leading_shape = xc.shape[:-1]

        # Step 2: squeeze to 6 numbers, turn them into angles between -pi and pi
        angles = math.pi * torch.tanh(self.squeeze(xc))
        flat_angles = angles.reshape(-1, self.number_of_qubits)

        # Step 3: run the circuit, expand back to the gate size, restore the leading shape
        measured = self.circuit(flat_angles)
        expanded = self.expand(measured)
        return expanded.reshape(*leading_shape, expanded.shape[-1])


# ==========================================================================================================
# Helpers
# ==========================================================================================================
def add_quantum_selection(model, number_of_qubits: int, number_of_layers: int) -> int:
    """Replace x_proj in every Chain-Mamba layer that has memory. Returns how many layers were changed."""
    changed = 0
    for layer in model.core.layers:
        if getattr(layer, "mixer", False):
            input_size = layer.x_proj.in_features
            output_size = layer.x_proj.out_features
            layer.x_proj = QuantumSelection(input_size, output_size, number_of_qubits, number_of_layers)
            changed = changed + 1
    return changed


def get_circuits(model) -> list:
    circuits = []
    for module in model.modules():
        if isinstance(module, QuantumCircuitSimulator):
            circuits.append(module)
    return circuits


def set_hardware(model, shots=None, gate_error: float = 0.0, readout_a: float = 0.0, readout_b: float = 0.0) -> None:
    """Set the same hardware defects on every circuit of the model (does nothing for a classical model)."""
    for circuit in get_circuits(model):
        circuit.shots = shots
        circuit.gate_error = float(gate_error)
        circuit.readout_a = float(readout_a)
        circuit.readout_b = float(readout_b)


def set_random_training_defects(model, python_random: random.Random, ranges: dict) -> None:
    """'Practising with noise': random defects for one training batch, inside the ranges from the config."""
    shots = python_random.choice(ranges["shots_choices"])
    gate_error = python_random.uniform(0.0, ranges["gate_error_max"])
    readout_a = python_random.uniform(0.0, ranges["readout_a_max"])
    readout_b = python_random.uniform(0.0, ranges["readout_b_max"])
    set_hardware(model, shots=shots, gate_error=gate_error, readout_a=readout_a, readout_b=readout_b)


def hardware_at_step(condition: str, step: int, total_steps: int, switch_every: int) -> dict:
    """
    The hardware defects of one test condition at one control step.
    sun_shadow_switching and combined_realistic switch every `switch_every` steps (sunlight first).
    """
    in_sunlight = (step // switch_every) % 2 == 0
    if condition == "perfect":
        return {}
    if condition == "measurement_100_shots":
        return {"shots": 100}
    if condition == "measurement_10_shots":
        return {"shots": 10}
    if condition == "gate_errors_moderate":
        return {"gate_error": 0.05}
    if condition == "gate_errors_severe":
        return {"gate_error": 0.15}
    if condition == "readout_bias_moderate":
        return {"readout_a": 0.02, "readout_b": 0.08}
    if condition == "readout_bias_severe":
        return {"readout_a": 0.02, "readout_b": 0.20}
    if condition == "drift_during_run":
        fraction = step / max(1, total_steps - 1)
        return {"readout_a": 0.02, "readout_b": 0.20 * fraction}
    if condition == "sun_shadow_switching":
        if in_sunlight:
            return {"readout_a": 0.02, "readout_b": 0.20}
        return {"readout_a": 0.02, "readout_b": 0.02}
    if condition == "combined_realistic":
        if in_sunlight:
            return {"shots": 100, "gate_error": 0.05, "readout_a": 0.02, "readout_b": 0.20}
        return {"shots": 100, "gate_error": 0.05, "readout_a": 0.02, "readout_b": 0.08}
    raise ValueError(f"Unknown hardware condition '{condition}'.")


@torch.no_grad()
def check_readout_formula() -> float:
    """
    Compare the fast readout-error formula with a slow exact calculation (a confusion matrix applied to
    every bitstring), on a small 3-qubit circuit. Returns the largest difference (should be below 1e-5).
    """
    generator_state = torch.get_rng_state()
    torch.manual_seed(1)
    circuit = QuantumCircuitSimulator(3, 2)
    angles = torch.empty(5, 3).uniform_(-math.pi, math.pi)
    torch.set_rng_state(generator_state)
    a_value = 0.03
    b_value = 0.12

    # Slow exact version: probabilities of every bitstring, then the misreading applied to each qubit
    samples = angles.shape[0]
    state = torch.zeros(samples, 8)
    state[:, 0] = 1.0
    for layer in range(2):
        for qubit in range(3):
            angle = circuit.input_scaling[layer, qubit] * angles[:, qubit] + circuit.layer_angles[layer, qubit]
            state = circuit.apply_ry(state, qubit, angle)
        state = state[:, circuit.cnot_ring_permutation]
    for qubit in range(3):
        state = circuit.apply_ry(state, qubit, circuit.final_angles[qubit].expand(samples))
    probabilities = state * state
    one_qubit = torch.tensor([[1.0 - a_value, b_value], [a_value, 1.0 - b_value]])   # columns: true 0, true 1
    confusion = torch.kron(torch.kron(one_qubit, one_qubit), one_qubit)
    slow = (probabilities @ confusion.T) @ circuit.observables

    # Fast formula
    circuit.readout_a = a_value
    circuit.readout_b = b_value
    fast = circuit(angles)
    return float((fast - slow).abs().max())
