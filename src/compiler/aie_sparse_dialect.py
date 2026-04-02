"""Minimal Python-side description for `aie_sparse` dialect constructs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


DIALECT_NAME = "aie_sparse"
SPMV_OP_NAME = "aie_sparse.spmv"


@dataclass
class AIESparseSpMVOp:
    """IR-side logical op for sparse matrix-vector multiplication.

    This class is a lightweight descriptor usable by compiler passes that
    operate on MLIR text and need a structured representation.
    """

    matrix: str
    vector_in: str
    vector_out: str
    num_rows: str
    attrs: Optional[Dict[str, str]] = None

    def to_mlir(self) -> str:
        attrs = self.attrs or {}
        attr_parts = [f"{k} = {v}" for k, v in attrs.items()]
        attr_text = " {" + ", ".join(attr_parts) + "}" if attr_parts else ""
        return (
            f"{SPMV_OP_NAME} {self.matrix}, {self.vector_in}, {self.vector_out}, {self.num_rows}"
            f"{attr_text}"
        )


def dialect_summary() -> str:
    return (
        "Dialect: aie_sparse\n"
        "Operation: aie_sparse.spmv\n"
        "Semantics: vec_out = sparse_matrix * vec_in (TCSR-backed kernel call)\n"
    )
