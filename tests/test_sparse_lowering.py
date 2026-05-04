"""Sparse lowering tests for TCSR/CSR backend payload rewriting."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backend.sparse_to_aie import SparseToAIEPass
from src.backend.aries_backend import ARIESBackend, ARIESBackendConfig


def _sample_sparse_mlir() -> str:
    return (
        'module attributes {aries.sparse_matrix = "[[1.0,0.0],[0.0,2.0]]"} {\n'
        '  func.func @main(%vec_in: memref<?xf32>, %vec_out: memref<?xf32>) {\n'
        '    %tmp = linalg.matmul\n'
        '    return\n'
        '  }\n'
        '}\n'
    )


def test_sparse_lowering_csr_injects_csr_kernel_call() -> None:
    lowering = SparseToAIEPass(sparse_format="csr")
    out = lowering.run(_sample_sparse_mlir())

    assert out.changed is True
    assert out.csr is not None
    assert out.tcsr is None
    assert "@spmv_csr" in out.transformed_mlir
    assert "@csr_values" in out.transformed_mlir
    assert "aries.csr" in out.transformed_mlir


def test_sparse_lowering_tcsr_injects_tcsr_kernel_call() -> None:
    lowering = SparseToAIEPass(sparse_format="tcsr")
    out = lowering.run(_sample_sparse_mlir())

    assert out.changed is True
    assert out.tcsr is not None
    assert out.csr is None
    assert "@spmv_tcsr" in out.transformed_mlir
    assert "@tcsr_values" in out.transformed_mlir
    assert "aries.tcsr" in out.transformed_mlir


def test_sparse_lowering_csr_accepts_direct_csr_payload() -> None:
    mlir = (
        'module attributes {aries.csr_input = "{\\"values\\":[1.0,2.0],\\"col_indices\\":[0,1],\\"row_ptr\\":[0,1,2],\\"shape\\":[2,2]}"} {\n'
        '  func.func @main(%vec_in: memref<?xf32>, %vec_out: memref<?xf32>) {\n'
        '    %tmp = linalg.matmul\n'
        '    return\n'
        '  }\n'
        '}\n'
    )

    lowering = SparseToAIEPass(sparse_format="csr")
    out = lowering.run(mlir)

    assert out.changed is True
    assert out.csr is not None
    assert out.csr.shape == (2, 2)
    assert "@spmv_csr" in out.transformed_mlir
    assert "@csr_values" in out.transformed_mlir


def test_aries_backend_optimize_honors_call_time_sparse_format(monkeypatch) -> None:
    backend = ARIESBackend(
        ARIESBackendConfig(enable_sparse_lowering=True, sparse_format="tcsr")
    )

    # Avoid external tool dependency; inspect MLIR that would be passed to aries-opt.
    monkeypatch.setattr(ARIESBackend, "_resolve_tool", lambda self, name: name)
    monkeypatch.setattr(ARIESBackend, "_run", lambda self, cmd, mlir_input, timeout_sec: mlir_input)

    out = backend.optimize(_sample_sparse_mlir(), sparse_format="csr")

    assert "@spmv_csr" in out
    assert "@spmv_tcsr" not in out
    assert "aries.csr" in out
