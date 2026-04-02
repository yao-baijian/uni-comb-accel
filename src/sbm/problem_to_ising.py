"""Problem-to-Ising conversion helpers for SBM-style solvers."""

from __future__ import annotations

import numpy as np
import torch


def maxcut_to_bsb(J: torch.Tensor) -> torch.Tensor:
    """Convert a MaxCut coupling matrix into a balanced SBM matrix."""

    return -0.5 * J


def bmincut_to_bsb(J: torch.Tensor, lambda_balance: float = 1.0) -> torch.Tensor:
    """Convert a balanced minimum cut instance into a balanced SBM matrix."""

    N = J.shape[0]
    ones = torch.ones(N, device=J.device, dtype=J.dtype)
    return -0.5 * J - 2.0 * lambda_balance * torch.outer(ones, ones)


def qblib_to_bsb(Q_orig, b_orig, num_vars: int, device):
    """Convert a QPLIB quadratic problem into an SBM-friendly Ising matrix."""

    Q = torch.tensor(Q_orig, dtype=torch.float32, device=device)
    Q_sym = 0.5 * (Q + Q.T)

    b = torch.tensor(b_orig, dtype=torch.float32, device=device)
    ones = torch.ones(num_vars, dtype=torch.float32, device=device)

    J_ising = 0.125 * Q_sym
    h_ising = 0.25 * torch.matmul(Q_sym, ones) + 0.5 * b

    J_tensor = torch.zeros((num_vars + 1, num_vars + 1), device=device)
    J_tensor[:num_vars, :num_vars] = J_ising
    J_tensor[:num_vars, num_vars] = 0.7 * h_ising
    J_tensor[num_vars, :num_vars] = 0.7 * h_ising

    return J_tensor, num_vars + 1


def tsp_to_hamiltonian(city_distances: np.ndarray, fixed_start_city: int = None) -> np.ndarray:
    """Build the TSP Ising Hamiltonian used by the TSP SBM solver."""

    N = len(city_distances)
    total_spins = N * N

    J = np.zeros((total_spins, total_spins))
    h = np.zeros(total_spins)

    for j in range(N):
        for i1 in range(N):
            idx1 = i1 * N + j
            for i2 in range(i1 + 1, N):
                idx2 = i2 * N + j
                J[idx1, idx2] += 2.0
                J[idx2, idx1] += 2.0
            h[idx1] += 2 * (N - 2)

    for i in range(N):
        for j1 in range(N):
            idx1 = i * N + j1
            for j2 in range(j1 + 1, N):
                idx2 = i * N + j2
                J[idx1, idx2] += 2.0
                J[idx2, idx1] += 2.0
            h[idx1] += 2 * (N - 2)

    max_distance = np.max(city_distances)
    w = 0.5 / (N * max_distance) if N * max_distance > 0 else 0.01
    w = min(w, 0.1)

    for j in range(N):
        next_j = (j + 1) % N
        for i1 in range(N):
            for i2 in range(N):
                if i1 != i2:
                    dist = city_distances[i1, i2]
                    idx1 = i1 * N + j
                    idx2 = i2 * N + next_j

                    J[idx1, idx2] += w * dist / 4
                    J[idx2, idx1] += w * dist / 4
                    h[idx1] += w * dist / 4
                    h[idx2] += w * dist / 4

    if fixed_start_city is not None:
        constraint_strength = 10.0
        for i in range(N):
            idx = i * N + 0
            if i == fixed_start_city:
                h[idx] -= constraint_strength
            else:
                h[idx] += constraint_strength

    J_combined = J.copy()
    for i in range(total_spins):
        J_combined[i, i] += h[i]

    return J_combined