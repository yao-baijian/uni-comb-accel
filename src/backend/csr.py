"""CSR utilities for sparse matrix preprocessing and kernel payload emission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sp = None


@dataclass
class CSRData:
    """Container for CSR data prepared for SpMV kernels."""

    values: np.ndarray
    col_indices: np.ndarray
    row_ptr: np.ndarray
    shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "col_indices": self.col_indices.tolist(),
            "row_ptr": self.row_ptr.tolist(),
            "shape": list(self.shape),
        }

    def to_c_arrays(self, prefix: str = "spmv") -> str:
        values = ", ".join(f"{float(v):.8g}f" for v in self.values)
        cols = ", ".join(str(int(v)) for v in self.col_indices)
        row_ptr = ", ".join(str(int(v)) for v in self.row_ptr)

        return (
            f"// Auto-generated CSR constants for {prefix}\n"
            f"static const int {prefix}_num_rows = {self.shape[0]};\n"
            f"static const int {prefix}_num_cols = {self.shape[1]};\n"
            f"static const float {prefix}_values[] = {{{values}}};\n"
            f"static const int {prefix}_col_indices[] = {{{cols}}};\n"
            f"static const int {prefix}_row_ptr[] = {{{row_ptr}}};\n"
        )


def generate_csr(
    sparse_matrix: Any,
    dtype: np.dtype = np.float32,
) -> CSRData:
    """Convert a sparse/dense matrix into standard CSR format."""

    csr = _to_csr(sparse_matrix, dtype=dtype)
    rows, cols = csr.shape

    return CSRData(
        values=np.asarray(csr.data, dtype=dtype),
        col_indices=np.asarray(csr.indices, dtype=np.int32),
        row_ptr=np.asarray(csr.indptr, dtype=np.int32),
        shape=(int(rows), int(cols)),
    )


def _to_csr(matrix: Any, dtype: np.dtype):
    if sp is not None and sp.issparse(matrix):
        return matrix.tocsr().astype(dtype)

    arr = np.asarray(matrix, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError("Input matrix must be 2D")

    if sp is not None:
        return sp.csr_matrix(arr)

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for r in range(arr.shape[0]):
        nz = np.nonzero(arr[r])[0]
        for c in nz.tolist():
            indices.append(int(c))
            data.append(float(arr[r, c]))
        indptr.append(len(indices))

    class _CSRProxy:
        def __init__(self) -> None:
            self.shape = arr.shape
            self.indptr = np.asarray(indptr, dtype=np.int32)
            self.indices = np.asarray(indices, dtype=np.int32)
            self.data = np.asarray(data, dtype=dtype)

    return _CSRProxy()
