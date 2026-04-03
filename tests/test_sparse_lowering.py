"""Sparse lowering tests for TCSR/CSR backend payload rewriting."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backend.sparse_to_aie import SparseToAIEPass


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
