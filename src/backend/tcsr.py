"""TCSR (Tiled CSR) utilities for sparse matrix preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sp = None


@dataclass
class TCSRData:
    """Container for tiled CSR data prepared for AIE kernels."""

    values: np.ndarray
    col_indices: np.ndarray
    row_ptr: np.ndarray
    tile_headers: np.ndarray
    shape: Tuple[int, int]
    tile_shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "col_indices": self.col_indices.tolist(),
            "row_ptr": self.row_ptr.tolist(),
            "tile_headers": self.tile_headers.tolist(),
            "shape": list(self.shape),
            "tile_shape": list(self.tile_shape),
        }

    def to_c_arrays(self, prefix: str = "spmv") -> str:
        """Emit C-style arrays for embedding into runtime/kernel glue code."""

        values = ", ".join(f"{float(v):.8g}f" for v in self.values)
        cols = ", ".join(str(int(v)) for v in self.col_indices)
        row_ptr = ", ".join(str(int(v)) for v in self.row_ptr)
        headers = ", ".join(str(int(v)) for v in self.tile_headers)

        header_count = len(self.tile_headers) // 7
        return (
            f"// Auto-generated TCSR constants for {prefix}\n"
            f"static const int {prefix}_num_rows = {self.shape[0]};\n"
            f"static const int {prefix}_num_cols = {self.shape[1]};\n"
            f"static const int {prefix}_tile_rows = {self.tile_shape[0]};\n"
            f"static const int {prefix}_tile_cols = {self.tile_shape[1]};\n"
            f"static const int {prefix}_tile_count = {header_count};\n"
            f"static const int {prefix}_tile_headers[] = {{{headers}}};\n"
            f"static const float {prefix}_values[] = {{{values}}};\n"
            f"static const int {prefix}_col_indices[] = {{{cols}}};\n"
            f"static const int {prefix}_row_ptr[] = {{{row_ptr}}};\n"
        )


def generate_tcsr(
    sparse_matrix: Any,
    tile_rows: int = 32,
    tile_cols: int = 32,
    dtype: np.dtype = np.float32,
) -> TCSRData:
    """Convert a sparse/dense matrix into TCSR format.

    Tile header layout (7 ints per tile):
    `[tile_row, tile_col, row_start, row_end, value_offset, row_ptr_offset, nnz]`
    """

    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile_rows and tile_cols must be positive")

    csr = _to_csr(sparse_matrix, dtype=dtype)
    num_rows, num_cols = csr.shape

    values: List[float] = []
    col_indices: List[int] = []
    row_ptr: List[int] = []
    tile_headers: List[int] = []

    for tr in range(0, num_rows, tile_rows):
        r0 = tr
        r1 = min(tr + tile_rows, num_rows)

        for tc in range(0, num_cols, tile_cols):
            c0 = tc
            c1 = min(tc + tile_cols, num_cols)

            tile_values: List[float] = []
            tile_col_indices: List[int] = []
            tile_row_ptr: List[int] = [0]

            for r in range(r0, r1):
                start = int(csr.indptr[r])
                end = int(csr.indptr[r + 1])

                row_nnz = 0
                for idx in range(start, end):
                    c = int(csr.indices[idx])
                    if c0 <= c < c1:
                        tile_values.append(float(csr.data[idx]))
                        tile_col_indices.append(c)
                        row_nnz += 1

                tile_row_ptr.append(tile_row_ptr[-1] + row_nnz)

            tile_nnz = tile_row_ptr[-1]

            if tile_nnz > 0:
                tile_value_offset = len(values)
                tile_row_ptr_offset = len(row_ptr)
                values.extend(tile_values)
                col_indices.extend(tile_col_indices)
                row_ptr.extend(tile_row_ptr)
                tile_headers.extend(
                    [
                        r0 // tile_rows,
                        c0 // tile_cols,
                        r0,
                        r1,
                        tile_value_offset,
                        tile_row_ptr_offset,
                        tile_nnz,
                    ]
                )

    if not row_ptr:
        row_ptr = [0]

    return TCSRData(
        values=np.asarray(values, dtype=dtype),
        col_indices=np.asarray(col_indices, dtype=np.int32),
        row_ptr=np.asarray(row_ptr, dtype=np.int32),
        tile_headers=np.asarray(tile_headers, dtype=np.int32),
        shape=(int(num_rows), int(num_cols)),
        tile_shape=(int(tile_rows), int(tile_cols)),
    )


def _to_csr(matrix: Any, dtype: np.dtype) -> Any:
    if sp is not None and sp.issparse(matrix):
        return matrix.tocsr().astype(dtype)

    arr = np.asarray(matrix, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError("Input matrix must be 2D")

    if sp is not None:
        return sp.csr_matrix(arr)

    # SciPy is unavailable: create a minimal CSR-like object from dense ndarray.
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
