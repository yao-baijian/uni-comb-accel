"""Autodiff frontend tests: JAX gradients + MLIR generation smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    _HAS_JAX = True
except Exception:
    # Fallback: ARIES submodule may carry a dedicated Python env with JAX.
    aries_site_candidates = sorted((REPO_ROOT / "tools" / "ARIES" / "aries" / "lib").glob("python*/site-packages"))
    for site in aries_site_candidates:
        if str(site) not in sys.path:
            sys.path.insert(0, str(site))
        try:
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore

            _HAS_JAX = True
            break
        except Exception:
            continue
    else:
        jax = None
        jnp = None
        _HAS_JAX = False

from src.compiler.autodiff import combine_modules, get_forward_backward_mlir, module_to_text


pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX is not installed in current env")


def _assert_mlir_has_core_structure(text: str) -> None:
    assert "module {" in text
    assert "func.func @forward" in text or "func.func @backward" in text


def test_scalar_square_grad() -> None:
    def f(x):
        return x * x

    x = jnp.array(3.0, dtype=jnp.float32)
    g = jax.grad(f)(x)
    np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * x), rtol=1e-6, atol=1e-6)

    fwd, bwd = get_forward_backward_mlir(f, (x,))
    merged = combine_modules(fwd, bwd)
    text = module_to_text(merged)
    _assert_mlir_has_core_structure(text)


def test_vector_sum_square_grad() -> None:
    def f(x):
        return jnp.sum(x * x)

    x = jnp.array([1.0, -2.0, 3.5, -4.0], dtype=jnp.float32)
    g = jax.grad(f)(x)
    np.testing.assert_allclose(np.asarray(g), np.asarray(2.0 * x), rtol=1e-5, atol=1e-5)

    fwd, bwd = get_forward_backward_mlir(f, (x,))
    text = module_to_text(combine_modules(fwd, bwd))
    _assert_mlir_has_core_structure(text)
    assert "linalg.generic" in text or "arith.mulf" in text


def test_matrix_matmul_grad() -> None:
    def f(X, Y):
        return jnp.sum(X @ Y)

    X = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    Y = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0

    gx, gy = jax.grad(f, argnums=(0, 1))(X, Y)
    # Analytical gradients for sum(X @ Y):
    # d/dX = ones(2,4) @ Y^T, d/dY = X^T @ ones(2,4)
    ones = jnp.ones((2, 4), dtype=jnp.float32)
    gx_ref = ones @ Y.T
    gy_ref = X.T @ ones
    np.testing.assert_allclose(np.asarray(gx), np.asarray(gx_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(gy), np.asarray(gy_ref), rtol=1e-5, atol=1e-5)

    fwd, bwd = get_forward_backward_mlir(f, (X, Y))
    text = module_to_text(combine_modules(fwd, bwd))
    _assert_mlir_has_core_structure(text)
    assert "linalg.matmul" in text


def test_maxcut_energy_grad() -> None:
    def maxcut_energy(s, W):
        outer = s[:, None] * s[None, :]
        return -0.5 * jnp.sum(W * outer)

    n = 8
    key = jax.random.PRNGKey(0)
    key_s, key_w = jax.random.split(key)
    s = jax.random.normal(key_s, (n,), dtype=jnp.float32)
    W = jax.random.uniform(key_w, (n, n), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    W = 0.5 * (W + W.T)

    gs, gW = jax.grad(maxcut_energy, argnums=(0, 1))(s, W)

    # Basic sanity: finite gradients and expected shapes.
    assert gs.shape == s.shape
    assert gW.shape == W.shape
    assert np.isfinite(np.asarray(gs)).all()
    assert np.isfinite(np.asarray(gW)).all()

    fwd, bwd = get_forward_backward_mlir(maxcut_energy, (s, W))
    text = module_to_text(combine_modules(fwd, bwd))
    _assert_mlir_has_core_structure(text)
