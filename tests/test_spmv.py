"""SpMV correctness and performance smoke test.

This benchmark validates TCSR conversion and a CPU simulation of the AIE SpMV
kernel path. If AIE simulator binaries are available, a hook point is left for
future integration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backend.tcsr import TCSRData, generate_tcsr

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None


def _build_random_sparse(m: int, n: int, density: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if sp is not None:
        mat = sp.random(m, n, density=density, format="csr", dtype=np.float32, random_state=seed)
        # Keep values well-conditioned for numerical comparison.
        mat.data = (mat.data * 2.0 - 1.0).astype(np.float32)
        x = rng.standard_normal(n, dtype=np.float32)
        return mat, x

    dense = np.zeros((m, n), dtype=np.float32)
    nnz = int(m * n * density)
    rows = rng.integers(0, m, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    vals = (rng.random(nnz, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    dense[rows, cols] = vals
    x = rng.standard_normal(n, dtype=np.float32)
    return dense, x


def _cpu_ref_spmv(matrix, x: np.ndarray) -> np.ndarray:
    if sp is not None and sp.issparse(matrix):
        return np.asarray(matrix.dot(x), dtype=np.float32)
    return np.asarray(matrix @ x, dtype=np.float32)


def _simulate_spmv_tcsr(tcsr: TCSRData, vec_in: np.ndarray) -> np.ndarray:
    rows = tcsr.shape[0]
    out = np.zeros(rows, dtype=np.float32)

    vals = tcsr.values
    cols = tcsr.col_indices
    ptr = tcsr.row_ptr
    headers = tcsr.tile_headers.reshape(-1, 7) if tcsr.tile_headers.size > 0 else np.empty((0, 7), dtype=np.int32)

    for tile in headers:
        row_start = int(tile[2])
        row_end = int(tile[3])
        value_offset = int(tile[4])
        row_ptr_offset = int(tile[5])

        for r in range(row_start, row_end):
            local_r = r - row_start
            begin = int(ptr[row_ptr_offset + local_r])
            end = int(ptr[row_ptr_offset + local_r + 1])
            if end <= begin:
                continue

            idx_slice = slice(value_offset + begin, value_offset + end)
            out[r] += float(np.dot(vals[idx_slice], vec_in[cols[idx_slice]]))

    return out


def test_spmv_tcsr_correctness_and_perf() -> None:
    m, n, density = 1000, 1000, 0.01
    matrix, x = _build_random_sparse(m, n, density, seed=7)

    t0 = time.perf_counter()
    ref = _cpu_ref_spmv(matrix, x)
    t1 = time.perf_counter()

    tcsr = generate_tcsr(matrix, tile_rows=32, tile_cols=32)

    t2 = time.perf_counter()
    sim = _simulate_spmv_tcsr(tcsr, x)
    t3 = time.perf_counter()

    np.testing.assert_allclose(sim, ref, rtol=1e-4, atol=1e-4)

    cpu_ms = (t1 - t0) * 1e3
    sim_ms = (t3 - t2) * 1e3
    print(f"[SpMV] CPU reference: {cpu_ms:.3f} ms")
    print(f"[SpMV] TCSR simulation: {sim_ms:.3f} ms")
    print("[SpMV] correctness check passed")