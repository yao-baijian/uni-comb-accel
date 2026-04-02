"""ARIES backend utilities for MLIR optimization and AIE code generation."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.backend.sparse_to_aie import SparseToAIEPass


@dataclass
class ARIESBackendConfig:
    """Configuration for ARIES command-line tool discovery and invocation."""

    project_root: Optional[Union[str, Path]] = None
    aries_root: Optional[Union[str, Path]] = None
    aries_opt: Optional[Union[str, Path]] = None
    aries_translate: Optional[Union[str, Path]] = None
    enable_sparse_lowering: bool = True
    sparse_tile_rows: int = 32
    sparse_tile_cols: int = 32


class ARIESBackend:
    """Run ARIES optimization and codegen from Python.

    Features:
    1. Accept MLIR text (or MLIR Python module-like object with `str(module)`).
    2. Run ARIES optimization passes via `aries-opt`.
    3. Run ARIES code generation via `aries-translate` or split passes.
    4. Support sparse operation to PE mapping via module-level attributes.
    """

    def __init__(
        self,
        config: Optional[ARIESBackendConfig] = None,
    ) -> None:
        self.config = config or ARIESBackendConfig()

        if self.config.project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(self.config.project_root).resolve()

        if self.config.aries_root is None:
            self.aries_root = self.project_root / "tools" / "ARIES"
        else:
            self.aries_root = Path(self.config.aries_root).resolve()

        self._sparse_pass = SparseToAIEPass(
            tile_rows=self.config.sparse_tile_rows,
            tile_cols=self.config.sparse_tile_cols,
        )

    def optimize(
        self,
        mlir_module: Union[str, Any],
        pipeline: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        sparse_mapping: Optional[Mapping[str, Union[int, Sequence[int], str]]] = None,
        timeout_sec: int = 120,
    ) -> str:
        """Run `aries-opt` on MLIR text and return optimized MLIR text.

        Args:
            mlir_module: MLIR text or module object.
            pipeline: Pipeline passed as `--pipeline=<pipeline>`.
            extra_args: Extra command args, for example `['-canonicalize']`.
            sparse_mapping: Optional sparse op -> PE mapping metadata.
            timeout_sec: Subprocess timeout.
        """

        mlir_text = self._coerce_mlir_text(mlir_module)

        if self.config.enable_sparse_lowering:
            sparse_result = self._sparse_pass.run(mlir_text)
            mlir_text = sparse_result.transformed_mlir

        if sparse_mapping:
            mlir_text = self._inject_sparse_mapping_attr(mlir_text, sparse_mapping)

        cmd: List[str] = [self._resolve_tool("aries-opt")]
        if pipeline:
            cmd.append(f"--pipeline={pipeline}")
        if extra_args:
            cmd.extend(extra_args)

        return self._run(cmd, mlir_text, timeout_sec=timeout_sec)

    def generate_aie_code(
        self,
        mlir_module: Union[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        emit: str = "cc",
        translate_args: Optional[Sequence[str]] = None,
        timeout_sec: int = 120,
    ) -> str:
        """Generate AIE kernel code from MLIR.

        Args:
            mlir_module: MLIR text or module object.
            output_path: Optional file path to write output (.cc or .mlir).
            emit: `cc`, `cpp`, `kernels`, or `mlir`.
            translate_args: Extra args for the selected command.
            timeout_sec: Subprocess timeout.

        Returns:
            Generated text (or resulting MLIR text for `emit='mlir'`).
        """

        mlir_text = self._coerce_mlir_text(mlir_module)

        if emit in {"cc", "cpp", "kernels"}:
            cmd = [self._resolve_tool("aries-translate")]
            if emit in {"cc", "cpp"}:
                # Matches ARIES template usage for kernel function C/C++ emission.
                cmd.append("-emit-kenrel-func")
            else:
                cmd.append("-emit-aries-kernels")
            if translate_args:
                cmd.extend(translate_args)
            generated = self._run(cmd, mlir_text, timeout_sec=timeout_sec)
        elif emit == "mlir":
            cmd = [self._resolve_tool("aries-opt"), "-aries-kernel-split"]
            if translate_args:
                cmd.extend(translate_args)
            generated = self._run(cmd, mlir_text, timeout_sec=timeout_sec)
        else:
            raise ValueError("emit must be one of: 'cc', 'cpp', 'kernels', 'mlir'")

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(generated, encoding="utf-8")

        return generated

    def annotate_sparse_mapping(
        self,
        mlir_module: Union[str, Any],
        mapping: Mapping[str, Union[int, Sequence[int], str]],
    ) -> str:
        """Attach sparse-to-PE mapping as module attribute metadata.

        The mapping is encoded as JSON string attribute:
        `aries.sparse_mapping = "{...}"`
        so downstream ARIES passes can inspect it.
        """

        mlir_text = self._coerce_mlir_text(mlir_module)
        return self._inject_sparse_mapping_attr(mlir_text, mapping)

    def optimize_and_codegen(
        self,
        mlir_module: Union[str, Any],
        pipeline: Optional[str] = None,
        opt_args: Optional[Sequence[str]] = None,
        sparse_mapping: Optional[Mapping[str, Union[int, Sequence[int], str]]] = None,
        emit: str = "cc",
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Convenience API: optimize MLIR, then generate AIE code."""

        optimized = self.optimize(
            mlir_module=mlir_module,
            pipeline=pipeline,
            extra_args=opt_args,
            sparse_mapping=sparse_mapping,
        )
        return self.generate_aie_code(
            mlir_module=optimized,
            output_path=output_path,
            emit=emit,
        )

    def _resolve_tool(self, tool_name: str) -> str:
        explicit = None
        if tool_name == "aries-opt" and self.config.aries_opt is not None:
            explicit = str(self.config.aries_opt)
        elif tool_name == "aries-translate" and self.config.aries_translate is not None:
            explicit = str(self.config.aries_translate)

        candidates: List[str] = []
        if explicit:
            candidates.append(explicit)

        candidates.extend(
            [
                str(self.aries_root / "build" / "bin" / tool_name),
                str(self.aries_root / "llvm-build" / "bin" / tool_name),
                tool_name,
            ]
        )

        for item in candidates:
            p = Path(item)
            if p.is_absolute() and p.exists():
                return str(p)
            if which(item) is not None:
                return item

        searched = "\n  - ".join(candidates)
        raise FileNotFoundError(
            f"Cannot find '{tool_name}'. Searched:\n  - {searched}\n"
            "Build ARIES first or provide explicit tool paths in ARIESBackendConfig."
        )

    def _run(self, cmd: Sequence[str], mlir_input: str, timeout_sec: int) -> str:
        try:
            proc = subprocess.run(
                list(cmd),
                input=mlir_input,
                text=True,
                capture_output=True,
                cwd=str(self.project_root),
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Command timed out after {timeout_sec}s: {shlex.join(cmd)}") from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                "ARIES command failed.\n"
                f"Command: {shlex.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"stderr:\n{stderr}"
            )

        return proc.stdout

    def _coerce_mlir_text(self, mlir_module: Union[str, Any]) -> str:
        if isinstance(mlir_module, str):
            text = mlir_module
        else:
            text = str(mlir_module)
        text = text.strip()
        if not text:
            raise ValueError("Empty MLIR module text")
        return text + "\n"

    def _inject_sparse_mapping_attr(
        self,
        mlir_text: str,
        mapping: Mapping[str, Union[int, Sequence[int], str]],
    ) -> str:
        lines = mlir_text.splitlines()
        if not lines:
            raise ValueError("Empty MLIR module text")

        payload = json.dumps(dict(mapping), separators=(",", ":"))
        payload = payload.replace('"', '\\"')

        # If a module attribute line already exists, update it in-place.
        for i, line in enumerate(lines):
            if line.strip().startswith("module attributes {"):
                if "aries.sparse_mapping" in line:
                    return mlir_text
                marker = "} {"
                pos = line.find(marker)
                if pos == -1:
                    raise ValueError(
                        "Unsupported `module attributes` layout for sparse mapping injection"
                    )
                before = line[:pos].rstrip()
                after = line[pos:]
                if before.endswith("{"):
                    new_before = f"{before} aries.sparse_mapping = \"{payload}\""
                else:
                    new_before = f"{before}, aries.sparse_mapping = \"{payload}\""
                lines[i] = f"{new_before}{after}"
                return "\n".join(lines) + "\n"

        # Canonical expected first line: `module {`
        for i, line in enumerate(lines):
            if line.strip() == "module {":
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = (
                    f"{indent}module attributes {{"
                    f"aries.sparse_mapping = \"{payload}\""
                    "} {"
                )
                return "\n".join(lines) + "\n"

        raise ValueError("Expected MLIR text to contain a top-level `module {` block")
