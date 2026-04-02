"""Benchmark FEM manual gradients vs JAX autodiff on FEM test instances.

This test avoids importing FEM package entrypoints (which pull pandas/pyarrow)
by loading only `tools/FEM/FEM/problem.py` and reusing its manual gradient
functions directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:
    jax = None
    jnp = None
    _HAS_JAX = False

from src.compiler.autodiff import get_forward_backward_mlir

pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX is not installed")


def _load_problem_module():
    problem_path = REPO_ROOT / "tools" / "FEM" / "FEM" / "problem.py"
    spec = importlib.util.spec_from_file_location("fem_problem_module", problem_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FEM problem module from {problem_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fem_problem = _load_problem_module()


def _read_graph(file_path: Path, index_start: int = 0) -> Tuple[int, int, torch.Tensor]:
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        n, m = int(header[0]), int(header[1])
        J = torch.zeros((n, n), dtype=torch.float32)
        for _ in range(m):
            parts = f.readline().split()
            i, j = int(parts[0]) - index_start, int(parts[1]) - index_start
            w = float(parts[2]) if len(parts) == 3 else 1.0
            J[i, j] = w
            J[j, i] = w
    return n, m, J


def _clause_mask_tensor(n: int, m: int, sat_table: List[List[List[int]]]) -> torch.Tensor:
    clause = []
    for ii in range(m):
        k = len(sat_table[ii][0])
        clause_m = torch.sparse_coo_tensor(
            [sat_table[ii][0], sat_table[ii][1]],
            [1] * k,
            (n, 2),
            dtype=torch.float32,
        ).to_dense().unsqueeze(0)
        clause.append(clause_m.unsqueeze(0))
    return torch.cat(clause, dim=0)  # [M, 1, N, 2]


def _read_cnf(path: Path) -> Tuple[int, int, torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sat_table = []
    n = m = 0
    for line in lines:
        l = line.split()
        if not l or l[0] == "c":
            continue
        if l[0] == "p":
            n, m = int(l[2]), int(l[3])
            continue
        clause = list(map(int, l[:-1]))
        nodes = [abs(x) - 1 for x in clause]
        states = [0 if x > 0 else 1 for x in clause]
        sat_table.append([nodes, states])

    real_m = len(sat_table)
    if real_m != m:
        raise ValueError(f"CNF clause count mismatch: header={m}, parsed={real_m}")
    return n, m, _clause_mask_tensor(n, m, sat_table)


def _betas(num_steps: int, betamin: float, betamax: float, anneal: str) -> np.ndarray:
    if anneal == "lin":
        return np.linspace(betamin, betamax, num_steps, dtype=np.float32)
    if anneal == "exp":
        return np.exp(np.linspace(np.log(betamin), np.log(betamax), num_steps)).astype(np.float32)
    if anneal == "inverse":
        return (1.0 / np.linspace(betamax, betamin, num_steps)).astype(np.float32)
    raise ValueError(f"Unknown anneal schedule: {anneal}")


def _entropy_grad_binary_torch(p: torch.Tensor) -> torch.Tensor:
    return -(p * (1 - p) * (torch.log(p + 1e-9) - torch.log(1 - p + 1e-9)))


def _entropy_grad_q_torch(p: torch.Tensor) -> torch.Tensor:
    tp = torch.log(p + 1e-9)
    return -p * (tp - (p * tp).sum(2, keepdim=True).expand_as(p))


def _entropy_binary_jax(p):
    return -jnp.sum(p * jnp.log(p + 1e-9) + (1.0 - p) * jnp.log(1.0 - p + 1e-9), axis=1)


def _entropy_q_jax(p):
    return -jnp.sum(p * jnp.log(p + 1e-9), axis=(1, 2))


def _maxksat_energy_jax(clause_mask, p):
    # clause_mask: [M, N, 2], p: [T, N, 2]
    term = 1.0 - p[None, :, :, :]
    mask = clause_mask[:, None, :, :] > 0.5
    selected = jnp.where(mask, term, 1.0)
    unsat = jnp.prod(selected, axis=(2, 3))
    return jnp.sum(unsat, axis=0)


def _maxksat_value_numpy(clause_mask: np.ndarray, p: np.ndarray) -> np.ndarray:
    term = 1.0 - p[None, :, :, :]
    mask = clause_mask[:, None, :, :] > 0.5
    selected = np.where(mask, term, 1.0)
    unsat = np.prod(selected, axis=(2, 3))
    return np.sum(unsat, axis=0)


def _run_manual_and_autodiff(
    problem_name: str,
    num_steps: int,
    h_shape: Tuple[int, ...],
    betamin: float,
    betamax: float,
    anneal: str,
    lr: float,
    h_factor: float,
    manual_grad_fn: Callable[[torch.Tensor], torch.Tensor],
    value_fn: Callable[[np.ndarray], np.ndarray],
    jax_free_energy_fn: Callable,
    seed: int = 1,
) -> Tuple[float, float]:
    betas = _betas(num_steps, betamin, betamax, anneal)

    rng = np.random.default_rng(seed)
    h0 = (h_factor * rng.standard_normal(size=h_shape)).astype(np.float32)

    # Manual-gradient path.
    h_manual = torch.tensor(h0, dtype=torch.float32)
    for beta in betas:
        if h_manual.ndim == 2:
            p_t = torch.sigmoid(h_manual)
            grad_t = manual_grad_fn(p_t) - _entropy_grad_binary_torch(p_t) / float(beta)
        else:
            p_t = torch.softmax(h_manual, dim=2)
            grad_t = manual_grad_fn(p_t) - _entropy_grad_q_torch(p_t) / float(beta)
        h_manual = h_manual - lr * grad_t

    if h_manual.ndim == 2:
        p_manual = torch.sigmoid(h_manual).detach().cpu().numpy()
    else:
        p_manual = torch.softmax(h_manual, dim=2).detach().cpu().numpy()
    v_manual = value_fn(p_manual)

    # Autodiff path.
    h_auto = jnp.asarray(h0)
    grad_fn = jax.jit(jax.grad(lambda h, beta: jnp.sum(jax_free_energy_fn(h, beta)), argnums=0))

    for beta in betas:
        g = grad_fn(h_auto, jnp.asarray(beta, dtype=jnp.float32))
        h_auto = h_auto - lr * g

    p_auto = np.asarray(jax.nn.sigmoid(h_auto) if h_auto.ndim == 2 else jax.nn.softmax(h_auto, axis=2))
    v_auto = value_fn(p_auto)

    if problem_name == "maxcut":
        return float(np.max(v_manual)), float(np.max(v_auto))
    return float(np.min(v_manual)), float(np.min(v_auto))


def test_fem_problem_manual_vs_autodiff_close() -> None:
    instances = REPO_ROOT / "tools" / "FEM" / "tests" / "test_instances"

    # ---------------- maxcut ----------------
    n, _m, J = _read_graph(instances / "G1.txt", index_start=1)
    c = 1.0 / torch.abs(J).sum(1)
    Jc = (c.reshape(-1, 1) * J).numpy().astype(np.float32)

    def maxcut_manual_grad(p_t: torch.Tensor) -> torch.Tensor:
        return fem_problem.manual_grad_maxcut(torch.tensor(Jc), p_t, discretization=False)

    def maxcut_value(p_np: np.ndarray) -> np.ndarray:
        _, result = fem_problem.infer_maxcut(J, torch.tensor(p_np, dtype=torch.float32))
        return result.detach().cpu().numpy()

    def maxcut_free_energy(h, beta):
        p = jax.nn.sigmoid(h)
        energy = -jnp.sum((p @ Jc) * (1.0 - p), axis=1)
        return energy - _entropy_binary_jax(p) / beta

    # Autodiff/MLIR smoke: use a supported tensor contraction skeleton.
    get_forward_backward_mlir(
        lambda x, y: jnp.sum(x @ y),
        (
            jnp.zeros((8, 8), dtype=jnp.float32),
            jnp.zeros((8, 8), dtype=jnp.float32),
        ),
        optimize_contractions=True,
    )

    maxcut_manual, maxcut_auto = _run_manual_and_autodiff(
        "maxcut",
        num_steps=80,
        h_shape=(24, n),
        betamin=0.001,
        betamax=0.5,
        anneal="inverse",
        lr=0.08,
        h_factor=0.01,
        manual_grad_fn=maxcut_manual_grad,
        value_fn=maxcut_value,
        jax_free_energy_fn=maxcut_free_energy,
    )

    # ---------------- bmincut ----------------
    n2, _m2, J2 = _read_graph(instances / "karate.txt", index_start=1)
    imba = float(5.0 * J2.square().sum().item() / (n2**2))
    J2_np = J2.numpy().astype(np.float32)

    def bmincut_manual_grad(p_t: torch.Tensor) -> torch.Tensor:
        return fem_problem.manual_grad_bmincut(torch.tensor(J2_np), p_t, imba)

    def bmincut_value(p_np: np.ndarray) -> np.ndarray:
        _, result = fem_problem.infer_bmincut(torch.tensor(J2_np, dtype=torch.float32), torch.tensor(p_np, dtype=torch.float32))
        return result.detach().cpu().numpy()

    def bmincut_free_energy(h, beta):
        p = jax.nn.softmax(h, axis=2)
        j = jnp.asarray(J2_np)
        exp_cut2 = jnp.sum((j @ p) * (1.0 - p), axis=(1, 2))
        penalty = jnp.sum((jnp.sum(p, axis=1) ** 2), axis=1) - jnp.sum(p * p, axis=(1, 2))
        energy = exp_cut2 + imba * penalty
        return energy - _entropy_q_jax(p) / beta

    get_forward_backward_mlir(
        lambda x, y, z: jnp.sum((x @ y) @ z),
        (
            jnp.zeros((4, 6), dtype=jnp.float32),
            jnp.zeros((6, 5), dtype=jnp.float32),
            jnp.zeros((5, 3), dtype=jnp.float32),
        ),
        optimize_contractions=True,
    )

    bmincut_manual, bmincut_auto = _run_manual_and_autodiff(
        "bmincut",
        num_steps=70,
        h_shape=(20, n2, 2),
        betamin=0.01,
        betamax=0.5,
        anneal="inverse",
        lr=0.05,
        h_factor=0.01,
        manual_grad_fn=bmincut_manual_grad,
        value_fn=bmincut_value,
        jax_free_energy_fn=bmincut_free_energy,
    )

    # ---------------- vertexcover ----------------
    n3, _m3, J3 = _read_graph(instances / "vertexcover.txt", index_start=0)
    deg = J3.sum(1)
    J3_mod = (J3 * (5.0 / 2.0)).clone()
    J3_mod[range(n3), range(n3)] = 1.0 - deg * 5.0
    J3_np = J3_mod.numpy().astype(np.float32)

    def vertex_manual_grad(p_t: torch.Tensor) -> torch.Tensor:
        return fem_problem.manual_grad_qubo(torch.tensor(J3_np), p_t)

    def vertex_value(p_np: np.ndarray) -> np.ndarray:
        _, result = fem_problem.infer_qubo(torch.tensor(J3_np, dtype=torch.float32), torch.tensor(p_np, dtype=torch.float32))
        return result.detach().cpu().numpy()

    def vertex_free_energy(h, beta):
        p = jax.nn.sigmoid(h)
        j = jnp.asarray(J3_np)
        energy = jnp.einsum("bi,ij,bj->b", p, j, p)
        return energy - _entropy_binary_jax(p) / beta

    get_forward_backward_mlir(
        lambda x, y: jnp.sum((x @ y) * (x @ y)),
        (
            jnp.zeros((8, 8), dtype=jnp.float32),
            jnp.zeros((8, 8), dtype=jnp.float32),
        ),
        optimize_contractions=True,
    )

    vertex_manual, vertex_auto = _run_manual_and_autodiff(
        "vertexcover",
        num_steps=70,
        h_shape=(20, n3),
        betamin=10.0,
        betamax=30.0,
        anneal="exp",
        lr=0.01,
        h_factor=1.0,
        manual_grad_fn=vertex_manual_grad,
        value_fn=vertex_value,
        jax_free_energy_fn=vertex_free_energy,
    )

    # ---------------- maxksat ----------------
    n4, _m4, clause_dense = _read_cnf(instances / "s3v70c1000-1.cnf")
    clause_dense_np = clause_dense.squeeze(1).numpy().astype(np.float32)  # [M, N, 2]
    clause_sparse_t = clause_dense.repeat(1, 20, 1, 1).to_sparse()

    def maxksat_manual_grad(p_t: torch.Tensor) -> torch.Tensor:
        return fem_problem.manual_grad_maxksat(clause_sparse_t, p_t.unsqueeze(0)).squeeze(0)

    def maxksat_value(p_np: np.ndarray) -> np.ndarray:
        config = np.eye(2, dtype=np.float32)[np.argmax(p_np, axis=2)]
        return _maxksat_value_numpy(clause_dense_np, config)

    clause_j = jnp.asarray(clause_dense_np)

    def maxksat_free_energy(h, beta):
        p = jax.nn.softmax(h, axis=2)
        energy = _maxksat_energy_jax(clause_j, p)
        return energy - _entropy_q_jax(p) / beta

    get_forward_backward_mlir(
        lambda a, b, c: jnp.sum((a @ b) @ c),
        (
            jnp.zeros((3, 4), dtype=jnp.float32),
            jnp.zeros((4, 5), dtype=jnp.float32),
            jnp.zeros((5, 2), dtype=jnp.float32),
        ),
        optimize_contractions=True,
    )

    maxksat_manual, maxksat_auto = _run_manual_and_autodiff(
        "maxksat",
        num_steps=60,
        h_shape=(20, n4, 2),
        betamin=0.01,
        betamax=30.0,
        anneal="lin",
        lr=0.08,
        h_factor=0.3,
        manual_grad_fn=maxksat_manual_grad,
        value_fn=maxksat_value,
        jax_free_energy_fn=maxksat_free_energy,
    )

    results: Dict[str, Tuple[float, float]] = {
        "maxcut": (maxcut_manual, maxcut_auto),
        "bmincut": (bmincut_manual, bmincut_auto),
        "vertexcover": (vertex_manual, vertex_auto),
        "maxksat": (maxksat_manual, maxksat_auto),
    }

    for name, (manual_best, auto_best) in results.items():
        denom = max(abs(manual_best), 1.0)
        rel_err = abs(auto_best - manual_best) / denom
        assert rel_err <= 0.01, (
            f"{name}: autodiff final value {auto_best:.6f} != manual {manual_best:.6f} "
            f"(rel err={rel_err:.4%})"
        )