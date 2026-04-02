"""Sparse format interface definitions for the backend and compiler."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class SparseFormat(str, Enum):
    TCSR = "tcsr"
    CSR = "csr"
    CSR5 = "csr5"
    BCOO = "bcoo"
    ORIGINAL_CSR = "original_csr"


def normalize_sparse_format(value: Optional[Union[str, SparseFormat]]) -> SparseFormat:
    if value is None:
        return SparseFormat.TCSR
    if isinstance(value, SparseFormat):
        return value

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "tcsr": SparseFormat.TCSR,
        "csr": SparseFormat.CSR,
        "csr5": SparseFormat.CSR5,
        "bcoo": SparseFormat.BCOO,
        "originalcsr": SparseFormat.ORIGINAL_CSR,
        "orig_csr": SparseFormat.ORIGINAL_CSR,
        "original_csr": SparseFormat.ORIGINAL_CSR,
    }
    if normalized in aliases:
        return aliases[normalized]

    try:
        return SparseFormat(normalized)
    except ValueError as exc:
        valid = ", ".join(item.value for item in SparseFormat)
        raise ValueError(f"Unsupported sparse format: {value!r}. Expected one of: {valid}") from exc