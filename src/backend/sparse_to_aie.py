"""Text-level sparse-to-AIE lowering helper.

This module rewrites sparse matmul-like MLIR snippets into an external call to
`spmv_tcsr` and injects TCSR constant payload metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.backend.csr import CSRData, generate_csr
from src.backend.tcsr import TCSRData, generate_tcsr
from src.backend.sparse_formats import SparseFormat, normalize_sparse_format


@dataclass
class SparseLoweringResult:
    transformed_mlir: str
    tcsr: Optional[TCSRData]
    csr: Optional[CSRData]
    changed: bool


class SparseToAIEPass:
    """Lower sparse matmul patterns to a kernel call site.

    Notes:
    - This is currently a pragmatic text-level pass intended to integrate with
      the Python-driven frontend flow.
    - It expects a marker attribute `aries.sparse_matrix = "<json-matrix>"` on
      the module or op, where the JSON decodes to a 2D matrix.
    """

    _SPARSE_OP_PATTERN = re.compile(
        r"(?P<ssa>%[A-Za-z0-9_]+)?\s*=\s*(sparse_tensor\.matmul|linalg\.matmul)\b"
    )
    _FUNC_HEADER_PATTERN = re.compile(r"^\s*func\.func\s+@", re.MULTILINE)

    def __init__(self, tile_rows: int = 32, tile_cols: int = 32, sparse_format: Optional[str] = None) -> None:
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.sparse_format = normalize_sparse_format(sparse_format)

    def run(self, mlir_text: str) -> SparseLoweringResult:
        if self.sparse_format not in {SparseFormat.TCSR, SparseFormat.CSR}:
            return SparseLoweringResult(
                transformed_mlir=mlir_text,
                tcsr=None,
                csr=None,
                changed=False,
            )

        if not self._SPARSE_OP_PATTERN.search(mlir_text):
            return SparseLoweringResult(
                transformed_mlir=mlir_text,
                tcsr=None,
                csr=None,
                changed=False,
            )

        matrix = self._extract_sparse_matrix_payload(mlir_text)
        if matrix is None:
            # No compile-time sparse payload. Keep original IR untouched.
            return SparseLoweringResult(
                transformed_mlir=mlir_text,
                tcsr=None,
                csr=None,
                changed=False,
            )

        if self.sparse_format == SparseFormat.CSR:
            csr = generate_csr(matrix)
            transformed = mlir_text
            transformed = self._inject_csr_metadata(transformed, csr)
            transformed = self._inject_csr_globals(transformed, csr)
            transformed = self._rewrite_sparse_ops_csr(transformed)
            transformed = self._ensure_spmv_decl_csr(transformed)
            return SparseLoweringResult(
                transformed_mlir=transformed,
                tcsr=None,
                csr=csr,
                changed=True,
            )

        tcsr = generate_tcsr(matrix, tile_rows=self.tile_rows, tile_cols=self.tile_cols)

        transformed = mlir_text
        transformed = self._inject_tcsr_metadata(transformed, tcsr)
        transformed = self._inject_tcsr_globals(transformed, tcsr)
        transformed = self._rewrite_sparse_ops(transformed)
        transformed = self._ensure_spmv_decl(transformed)

        return SparseLoweringResult(transformed_mlir=transformed, tcsr=tcsr, csr=None, changed=True)

    def _inject_csr_metadata(self, mlir_text: str, csr: CSRData) -> str:
        payload = json.dumps(csr.to_dict(), separators=(",", ":")).replace('"', '\\"')

        lines = mlir_text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("module attributes {"):
                if "aries.csr" in line:
                    return mlir_text
                marker = "} {"
                pos = line.find(marker)
                if pos < 0:
                    continue
                before = line[:pos].rstrip()
                after = line[pos:]
                if before.endswith("{"):
                    before = f"{before} aries.csr = \"{payload}\""
                else:
                    before = f"{before}, aries.csr = \"{payload}\""
                lines[i] = before + after
                return "\n".join(lines) + "\n"

            if stripped == "module {":
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}module attributes {{aries.csr = \"{payload}\"}} {{"
                return "\n".join(lines) + "\n"

        return mlir_text

    def _extract_sparse_matrix_payload(self, mlir_text: str) -> Optional[Any]:
        m = re.search(r'aries\.sparse_matrix\s*=\s*"(?P<payload>[^\"]+)"', mlir_text)
        if m is None:
            return None
        raw = m.group("payload")
        unescaped = raw.replace('\\"', '"')
        try:
            return json.loads(unescaped)
        except json.JSONDecodeError:
            return None

    def _inject_tcsr_metadata(self, mlir_text: str, tcsr: TCSRData) -> str:
        payload = json.dumps(tcsr.to_dict(), separators=(",", ":")).replace('"', '\\"')

        lines = mlir_text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("module attributes {"):
                if "aries.tcsr" in line:
                    return mlir_text
                marker = "} {"
                pos = line.find(marker)
                if pos < 0:
                    continue
                before = line[:pos].rstrip()
                after = line[pos:]
                if before.endswith("{"):
                    before = f"{before} aries.tcsr = \"{payload}\""
                else:
                    before = f"{before}, aries.tcsr = \"{payload}\""
                lines[i] = before + after
                return "\n".join(lines) + "\n"

            if stripped == "module {":
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}module attributes {{aries.tcsr = \"{payload}\"}} {{"
                return "\n".join(lines) + "\n"

        return mlir_text

    def _rewrite_sparse_ops(self, mlir_text: str) -> str:
        has_external_call = "aie.external_call" in mlir_text

        repl = (
            "%tcsr_th = memref.get_global @tcsr_tile_headers : memref<?xi32>\n"
            "%tcsr_val = memref.get_global @tcsr_values : memref<?xf32>\n"
            "%tcsr_ci = memref.get_global @tcsr_col_indices : memref<?xi32>\n"
            "%tcsr_rp = memref.get_global @tcsr_row_ptr : memref<?xi32>\n"
            "%tcsr_n = arith.constant -1 : i32\n"
        )
        if has_external_call:
            repl += (
                "aie.external_call @spmv_tcsr"
                "(%tcsr_th, %tcsr_val, %tcsr_ci, %tcsr_rp, %vec_in, %vec_out, %tcsr_n)"
            )
        else:
            repl += (
                "func.call @spmv_tcsr"
                "(%tcsr_th, %tcsr_val, %tcsr_ci, %tcsr_rp, %vec_in, %vec_out, %tcsr_n)"
                " : (memref<?xi32>, memref<?xf32>, memref<?xi32>, memref<?xi32>, "
                "memref<?xf32>, memref<?xf32>, i32) -> ()"
            )

        def _replace(match: re.Match[str]) -> str:
            return repl

        return self._SPARSE_OP_PATTERN.sub(_replace, mlir_text)

    def _inject_tcsr_globals(self, mlir_text: str, tcsr: TCSRData) -> str:
        if "memref.global \"private\" constant @tcsr_values" in mlir_text:
            return mlir_text

        values = ", ".join(f"{float(v):.8g}" for v in tcsr.values.tolist())
        col_indices = ", ".join(str(int(v)) for v in tcsr.col_indices.tolist())
        row_ptr = ", ".join(str(int(v)) for v in tcsr.row_ptr.tolist())
        tile_headers = ", ".join(str(int(v)) for v in tcsr.tile_headers.tolist())

        def _tensor_literal(data: str, n: int, typ: str) -> str:
            if n == 0:
                zero = "0.0" if typ.startswith("f") else "0"
                return f"dense<[{zero}]> : tensor<1x{typ}>"
            return f"dense<[{data}]> : tensor<{n}x{typ}>"

        globals_text = "\n".join(
            [
                "  memref.global \"private\" constant @tcsr_tile_headers"
                f" : memref<{max(len(tcsr.tile_headers), 1)}xi32> = "
                f"{_tensor_literal(tile_headers, len(tcsr.tile_headers), 'i32')}",
                "  memref.global \"private\" constant @tcsr_values"
                f" : memref<{max(len(tcsr.values), 1)}xf32> = "
                f"{_tensor_literal(values, len(tcsr.values), 'f32')}",
                "  memref.global \"private\" constant @tcsr_col_indices"
                f" : memref<{max(len(tcsr.col_indices), 1)}xi32> = "
                f"{_tensor_literal(col_indices, len(tcsr.col_indices), 'i32')}",
                "  memref.global \"private\" constant @tcsr_row_ptr"
                f" : memref<{max(len(tcsr.row_ptr), 1)}xi32> = "
                f"{_tensor_literal(row_ptr, len(tcsr.row_ptr), 'i32')}",
            ]
        )

        m = self._FUNC_HEADER_PATTERN.search(mlir_text)
        if m is None:
            lines = mlir_text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "module {" or line.strip().startswith("module attributes {"):
                    lines.insert(i + 1, globals_text)
                    return "\n".join(lines) + "\n"
            return mlir_text

        insert_pos = m.start()
        return mlir_text[:insert_pos] + globals_text + "\n" + mlir_text[insert_pos:]

    def _inject_csr_globals(self, mlir_text: str, csr: CSRData) -> str:
        if "memref.global \"private\" constant @csr_values" in mlir_text:
            return mlir_text

        values = ", ".join(f"{float(v):.8g}" for v in csr.values.tolist())
        col_indices = ", ".join(str(int(v)) for v in csr.col_indices.tolist())
        row_ptr = ", ".join(str(int(v)) for v in csr.row_ptr.tolist())

        def _tensor_literal(data: str, n: int, typ: str) -> str:
            if n == 0:
                zero = "0.0" if typ.startswith("f") else "0"
                return f"dense<[{zero}]> : tensor<1x{typ}>"
            return f"dense<[{data}]> : tensor<{n}x{typ}>"

        globals_text = "\n".join(
            [
                "  memref.global \"private\" constant @csr_values"
                f" : memref<{max(len(csr.values), 1)}xf32> = "
                f"{_tensor_literal(values, len(csr.values), 'f32')}",
                "  memref.global \"private\" constant @csr_col_indices"
                f" : memref<{max(len(csr.col_indices), 1)}xi32> = "
                f"{_tensor_literal(col_indices, len(csr.col_indices), 'i32')}",
                "  memref.global \"private\" constant @csr_row_ptr"
                f" : memref<{max(len(csr.row_ptr), 1)}xi32> = "
                f"{_tensor_literal(row_ptr, len(csr.row_ptr), 'i32')}",
            ]
        )

        m = self._FUNC_HEADER_PATTERN.search(mlir_text)
        if m is None:
            lines = mlir_text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "module {" or line.strip().startswith("module attributes {"):
                    lines.insert(i + 1, globals_text)
                    return "\n".join(lines) + "\n"
            return mlir_text

        insert_pos = m.start()
        return mlir_text[:insert_pos] + globals_text + "\n" + mlir_text[insert_pos:]

    def _rewrite_sparse_ops_csr(self, mlir_text: str) -> str:
        has_external_call = "aie.external_call" in mlir_text

        repl = (
            "%csr_val = memref.get_global @csr_values : memref<?xf32>\n"
            "%csr_ci = memref.get_global @csr_col_indices : memref<?xi32>\n"
            "%csr_rp = memref.get_global @csr_row_ptr : memref<?xi32>\n"
            "%csr_n = arith.constant -1 : i32\n"
        )
        if has_external_call:
            repl += (
                "aie.external_call @spmv_csr"
                "(%csr_val, %csr_ci, %csr_rp, %vec_in, %vec_out, %csr_n)"
            )
        else:
            repl += (
                "func.call @spmv_csr"
                "(%csr_val, %csr_ci, %csr_rp, %vec_in, %vec_out, %csr_n)"
                " : (memref<?xf32>, memref<?xi32>, memref<?xi32>, "
                "memref<?xf32>, memref<?xf32>, i32) -> ()"
            )

        def _replace(match: re.Match[str]) -> str:
            return repl

        return self._SPARSE_OP_PATTERN.sub(_replace, mlir_text)

    def _ensure_spmv_decl(self, mlir_text: str) -> str:
        if "func.func private @spmv_tcsr" in mlir_text:
            return mlir_text

        lines = mlir_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "module {" or line.strip().startswith("module attributes {"):
                lines.insert(
                    i + 1,
                    "  func.func private @spmv_tcsr("
                    "%tile_headers: memref<?xi32>, %values: memref<?xf32>,"
                    " %col_indices: memref<?xi32>, %row_ptr: memref<?xi32>,"
                    " %vec_in: memref<?xf32>, %vec_out: memref<?xf32>, %num_rows: i32) -> ()",
                )
                return "\n".join(lines) + "\n"
        return mlir_text

    def _ensure_spmv_decl_csr(self, mlir_text: str) -> str:
        if "func.func private @spmv_csr" in mlir_text:
            return mlir_text

        lines = mlir_text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "module {" or line.strip().startswith("module attributes {"):
                lines.insert(
                    i + 1,
                    "  func.func private @spmv_csr("
                    "%values: memref<?xf32>, %col_indices: memref<?xi32>,"
                    " %row_ptr: memref<?xi32>, %vec_in: memref<?xf32>,"
                    " %vec_out: memref<?xf32>, %num_rows: i32) -> ()",
                )
                return "\n".join(lines) + "\n"
        return mlir_text
