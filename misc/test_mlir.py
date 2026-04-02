#!/usr/bin/env python3
"""Simple MLIR Python API import check."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from mlir import ir
    except Exception as exc:
        print("[FAIL] Cannot import MLIR Python API: from mlir import ir")
        print(f"Reason: {exc}")
        print("Hint: install MLIR Python bindings in your active environment.")
        return 1

    try:
        with ir.Context() as ctx:
            _ = ir.Location.unknown(ctx)
        print("[OK] MLIR Python API import and basic context creation succeeded.")
        return 0
    except Exception as exc:
        print("[FAIL] MLIR imported but basic API usage failed.")
        print(f"Reason: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
