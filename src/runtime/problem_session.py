"""Two-step problem workflow: definition/compilation and solving/execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from src.api import compile_energy_function


@dataclass
class ProblemHandle:
    """Stable handle to a compiled problem definition."""

    problem_id: str
    problem_type: str
    problem_dir: str
    manifest_path: str
    artifacts: Dict[str, str]
    reused_artifacts: bool


class BoardRuntime:
    """Runtime interface for FPGA/board code-loading state."""

    def is_loaded(self, problem_id: str, artifacts: Optional[Dict[str, str]] = None) -> bool:  # pragma: no cover - interface only
        raise NotImplementedError

    def load(self, problem_id: str, artifacts: Dict[str, str]) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class InMemoryBoardRuntime(BoardRuntime):
    """Minimal board runtime for local workflows and tests."""

    def __init__(self) -> None:
        self._loaded: Dict[str, str] = {}

    def is_loaded(self, problem_id: str, artifacts: Optional[Dict[str, str]] = None) -> bool:
        expected = self._loaded.get(problem_id)
        if expected is None:
            return False
        if artifacts is None:
            return True
        return expected == _artifact_fingerprint(artifacts)

    def load(self, problem_id: str, artifacts: Dict[str, str]) -> None:
        self._loaded[problem_id] = _artifact_fingerprint(artifacts)


class FileBoardRuntime(BoardRuntime):
    """Persistent board runtime state backed by a JSON file.

    This implementation does not communicate with FPGA drivers directly. It
    persists the loaded-state bookkeeping so solve-time checks survive process
    restarts and can detect stale loads when artifacts change.
    """

    def __init__(self, state_file: str = "build/board_runtime/state.json") -> None:
        self.state_path = Path(state_file)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._read_state()

    def is_loaded(self, problem_id: str, artifacts: Optional[Dict[str, str]] = None) -> bool:
        board_entry = self._state.get(problem_id)
        if not isinstance(board_entry, dict):
            return False

        loaded = bool(board_entry.get("loaded", False))
        if not loaded:
            return False

        if artifacts is None:
            return True

        expected = board_entry.get("artifact_fingerprint")
        return expected == _artifact_fingerprint(artifacts)

    def load(self, problem_id: str, artifacts: Dict[str, str]) -> None:
        self._state[problem_id] = {
            "loaded": True,
            "artifact_fingerprint": _artifact_fingerprint(artifacts),
            "artifacts": dict(artifacts),
        }
        self._write_state()

    def unload(self, problem_id: str) -> None:
        if problem_id in self._state:
            self._state[problem_id]["loaded"] = False
            self._write_state()

    def _read_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(loaded, dict):
            return {}
        return loaded

    def _write_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")


class ProblemSessionManager:
    """Manages define/solve lifecycle with compile artifact caching.

    Key behavior:
    - `define_problem`: compile only when problem signature changes.
    - `solve_problem`: check whether code is loaded to board before solve.
    """

    def __init__(self, cache_dir: str = "build/problem_cache") -> None:
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def define_problem(
        self,
        problem_type: str,
        energy_function: Callable[..., Any],
        example_args,
        *,
        target: str = "aie",
        gradient_mode: str = "auto",
        sparse_format: str = "tcsr",
        precision: str = "fp32",
        use_omeinsum: bool = True,
        julia_cmd: str = "julia",
        variables=None,
        expected_input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        auto_aie_config: bool = True,
    ) -> ProblemHandle:
        """Define and (if needed) compile a problem.

        Reuses prebuilt AIE artifacts when the problem signature is unchanged.
        """

        signature = self._build_signature(
            problem_type=problem_type,
            energy_function=energy_function,
            target=target,
            gradient_mode=gradient_mode,
            sparse_format=sparse_format,
            precision=precision,
            expected_input_shapes=expected_input_shapes,
            auto_aie_config=auto_aie_config,
        )
        problem_id = self._hash_signature(signature)
        problem_dir = self.cache_root / problem_id
        manifest_path = problem_dir / "manifest.json"

        existing = self._load_manifest_if_valid(manifest_path)
        if existing is not None:
            return ProblemHandle(
                problem_id=problem_id,
                problem_type=problem_type,
                problem_dir=str(problem_dir),
                manifest_path=str(manifest_path),
                artifacts=dict(existing.get("artifacts", {})),
                reused_artifacts=True,
            )

        problem_dir.mkdir(parents=True, exist_ok=True)
        artifacts = compile_energy_function(
            energy_function,
            example_args,
            target=target,
            output_dir=str(problem_dir),
            use_omeinsum=use_omeinsum,
            julia_cmd=julia_cmd,
            gradient_mode=gradient_mode,
            variables=variables,
            sparse_format=sparse_format,
            precision=precision,
            expected_input_shapes=expected_input_shapes,
            auto_aie_config=auto_aie_config,
        )

        manifest = {
            "problem_id": problem_id,
            "problem_type": problem_type,
            "signature": signature,
            "artifacts": artifacts,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return ProblemHandle(
            problem_id=problem_id,
            problem_type=problem_type,
            problem_dir=str(problem_dir),
            manifest_path=str(manifest_path),
            artifacts=dict(artifacts),
            reused_artifacts=False,
        )

    def solve_problem(
        self,
        handle: ProblemHandle,
        solver_fn: Callable[..., Any],
        *solver_args,
        board_runtime: Optional[BoardRuntime] = None,
        auto_load_to_board: bool = False,
        require_board: bool = True,
        **solver_kwargs,
    ) -> Any:
        """Solve a previously-defined problem.

        This method is intentionally separate from `define_problem` so user code
        can reuse the same compiled artifacts and avoid AIE regen.
        """

        if require_board:
            if board_runtime is None:
                raise RuntimeError(
                    "Board runtime is required for solve step when require_board=True. "
                    "Pass a BoardRuntime implementation and load matching artifacts first."
                )

            if not board_runtime.is_loaded(handle.problem_id, handle.artifacts):
                if auto_load_to_board:
                    board_runtime.load(handle.problem_id, handle.artifacts)
                else:
                    raise RuntimeError(
                        "AIE code is not loaded on board for this problem definition "
                        "(or loaded artifacts are stale). "
                        "Call board_runtime.load(...) or use auto_load_to_board=True."
                    )

        return solver_fn(*solver_args, **solver_kwargs)

    def _load_manifest_if_valid(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            return None

        # Ensure all recorded artifact files still exist before reusing.
        for _, path in artifacts.items():
            if isinstance(path, str) and path:
                if not Path(path).exists():
                    return None

        return manifest

    def _build_signature(
        self,
        *,
        problem_type: str,
        energy_function: Callable[..., Any],
        target: str,
        gradient_mode: str,
        sparse_format: str,
        precision: str,
        expected_input_shapes: Optional[Mapping[str, Sequence[int]]],
        auto_aie_config: bool,
    ) -> Dict[str, Any]:
        try:
            source = inspect.getsource(energy_function)
        except Exception:
            source = repr(energy_function)

        return {
            "problem_type": str(problem_type),
            "function_name": getattr(energy_function, "__name__", "anonymous"),
            "function_source": source,
            "target": str(target),
            "gradient_mode": str(gradient_mode),
            "sparse_format": str(sparse_format),
            "precision": str(precision),
            "expected_input_shapes": {
                k: [int(x) for x in v]
                for k, v in (expected_input_shapes or {}).items()
            },
            "auto_aie_config": bool(auto_aie_config),
        }

    def _hash_signature(self, signature: Dict[str, Any]) -> str:
        payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _artifact_fingerprint(artifacts: Dict[str, str]) -> str:
    payload = json.dumps(dict(artifacts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
