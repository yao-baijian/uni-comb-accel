from __future__ import annotations

import numpy as np


def csr_to_segmented_ell(
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    vals: np.ndarray,
    seg0_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert CSR arrays to ARIES-friendly segmented ELL tensors.

    This keeps semantics of segmented vector banks without dynamic loop bounds
    or runtime branching inside the tile kernel.

    Returns:
      col_local:  [rows, max_nnz] int32 indices into X0
      val_local:  [rows, max_nnz] float32 values for X0 segment
      col_remote: [rows, max_nnz] int32 indices into X1 (already adjusted by -seg0_end)
      val_remote: [rows, max_nnz] float32 values for X1 segment

    Padding entries are zeros so kernels can iterate a fixed max_nnz loop.
    """
    rows = int(len(row_ptr) - 1)
    nnz_per_row = row_ptr[1:] - row_ptr[:-1]
    max_nnz = int(np.max(nnz_per_row)) if rows > 0 else 0

    col_local = np.zeros((rows, max_nnz), dtype=np.int32)
    val_local = np.zeros((rows, max_nnz), dtype=np.float32)
    col_remote = np.zeros((rows, max_nnz), dtype=np.int32)
    val_remote = np.zeros((rows, max_nnz), dtype=np.float32)

    for r in range(rows):
        s = int(row_ptr[r])
        e = int(row_ptr[r + 1])
        k = 0
        for p in range(s, e):
            c = int(col_idx[p])
            v = float(vals[p])
            if c < seg0_end:
                col_local[r, k] = c
                val_local[r, k] = v
            else:
                col_remote[r, k] = c - seg0_end
                val_remote[r, k] = v
            k += 1

    return col_local, val_local, col_remote, val_remote
