"""Compile pipeline tests: end-to-end compile smoke and shape-policy cache behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jax.numpy as jnp  # type: ignore

    _HAS_JAX = True
except Exception:
    jnp = None
    _HAS_JAX = False

from src.api import compile_energy_function
from src.runtime.problem_session import ProblemSessionManager


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.skipif(not _HAS_JAX, reason="JAX is not installed in current env")
def test_compile_energy_function_full_flow(tmp_path, monkeypatch) -> None:
    """Compile through autodiff + backend stages without external ARIES binaries."""

    def fake_optimize(self, mlir_module, **kwargs):
        return str(mlir_module) + "\n// fake-opt"

    def fake_codegen(self, mlir_module, output_path=None, emit="cc", **kwargs):
        generated = f"// fake-{emit}\n{mlir_module}"
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(generated, encoding="utf-8")
        return generated

    monkeypatch.setattr("src.backend.aries_backend.ARIESBackend.optimize", fake_optimize)
    monkeypatch.setattr("src.backend.aries_backend.ARIESBackend.generate_aie_code", fake_codegen)

    def energy(x):
        return jnp.sum(x * x)

    x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    artifacts = compile_energy_function(
        energy,
        (x,),
        target="aie",
        output_dir=str(tmp_path / "compile_flow"),
        shape_policy="exact",
    )

    assert Path(artifacts["forward_mlir"]).exists()
    assert Path(artifacts["backward_mlir"]).exists()
    assert Path(artifacts["combined_mlir"]).exists()
    assert Path(artifacts["optimized_mlir"]).exists()
    assert Path(artifacts["code"]).exists()

    actual_shapes = json.loads(artifacts["actual_input_shapes"])
    compile_shapes = json.loads(artifacts["compile_input_shapes"])
    assert actual_shapes == compile_shapes
    assert tuple(actual_shapes["x"]) == (3,)


def test_shape_policy_cache_hit_and_recompile(tmp_path, monkeypatch) -> None:
    """Different input shapes can hit cache within same policy config; changing policy config recompiles."""

    calls = {"count": 0}

    def fake_compile_energy_function(
        func,
        example_args,
        target="aie",
        output_dir="build",
        **kwargs,
    ):
        calls["count"] += 1
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / f"artifact_{calls['count']}.txt"
        marker.write_text("ok", encoding="utf-8")
        return {
            "optimized_mlir": str(marker),
            "code": str(marker),
            "target": target,
            "shape_policy": str(kwargs.get("shape_policy", "exact")),
        }

    monkeypatch.setattr(
        "src.runtime.problem_session.compile_energy_function",
        fake_compile_energy_function,
    )

    manager = ProblemSessionManager(cache_dir=str(tmp_path / "problem_cache"))

    def energy(x):
        return float(np.sum(np.asarray(x) ** 2))

    h1 = manager.define_problem(
        "quad",
        energy,
        example_args=(np.zeros((64,), dtype=np.float32),),
        shape_policy="bucket",
        shape_buckets={"x": [(64,), (128,)]},
    )
    assert calls["count"] == 1
    assert h1.reused_artifacts is False

    h2 = manager.define_problem(
        "quad",
        energy,
        example_args=(np.zeros((96,), dtype=np.float32),),
        shape_policy="bucket",
        shape_buckets={"x": [(64,), (128,)]},
    )
    assert calls["count"] == 1
    assert h2.reused_artifacts is True
    assert h2.problem_id == h1.problem_id

    h3 = manager.define_problem(
        "quad",
        energy,
        example_args=(np.zeros((96,), dtype=np.float32),),
        shape_policy="bucket",
        shape_buckets={"x": [(128,), (256,)]},
    )
    assert calls["count"] == 2
    assert h3.reused_artifacts is False
    assert h3.problem_id != h1.problem_id
