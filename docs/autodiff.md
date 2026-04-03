# Autodiff Model

## Scope

This document describes the autodiff frontend path that lowers user Python
energy functions to MLIR using JAX tracing.

Primary implementation file:

- `src/compiler/autodiff.py`

Integration entrypoint:

- `src/api.py` via `compile_energy_function(...)`

## Pipeline

The current pipeline is:

1. Trace forward graph with `jax.make_jaxpr(func)(*example_args)`.
2. Trace backward graph with `jax.grad(...)` + `make_jaxpr`.
3. Lower each JAXPR through `JaxprToMLIR` into `func.func`.
4. Merge forward/backward modules using `combine_modules(...)`.

## OMEinsum Integration

Optional contraction-order optimization is supported via Julia packages:

- `OMEinsum`
- `OMEinsumContractionOrders`
- `JSON`

Install in Julia:

```julia
using Pkg
Pkg.add("OMEinsum")
Pkg.add("OMEinsumContractionOrders")
Pkg.add("JSON")
```

Behavior:

- Detects pure 2D matrix-chain patterns (`A @ B @ C @ ...`).
- Queries Julia for pairwise contraction order.
- Falls back to Python dynamic-programming order if Julia side is unavailable.
- Emits optimized `linalg.matmul` sequence.

Current optimization scope is matrix-chain contractions; general einsum
hypergraph optimization is not fully lowered yet.

## Supported Primitives

- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Linear algebra: `dot_general`, `broadcast_in_dim`, `transpose`
- Math: `exp`, `log`, `sin`, `cos`
- Reduction: `reduce_sum` (skeleton), `reduce_max` (placeholder)
- Utility: `reshape`, `squeeze`, `convert_element_type`, `copy`

Unsupported primitives raise `NotImplementedError` with the primitive name.

## API Usage

Minimal usage:

```python
from src.api import compile_energy_function

artifacts = compile_energy_function(
	func,
	example_args,
	target="aie",
	output_dir="build",
)
```

With OMEinsum enabled (default):

```python
artifacts = compile_energy_function(
	func,
	example_args,
	target="aie",
	output_dir="build",
	use_omeinsum=True,
	julia_cmd="julia",
)
```

Common generated files:

- `<name>.forward.mlir`
- `<name>.backward.mlir`
- `<name>.combined.mlir`
- `<name>.combined.opt.mlir`

## Current Status

Implemented and available:

- forward/backward JAXPR tracing and MLIR generation
- module merge and backend handoff
- optional OMEinsum contraction order optimization

Known gaps:

- full general-einsum order optimization/lowering
- richer save-for-backward buffer materialization

## Extension Guide

To add a new primitive:

1. Extend dispatch in `JaxprToMLIR._lower_eqn`.
2. Add a lowering helper that emits MLIR ops.
3. Register output vars in internal maps.
4. Add tests in `tests/test_autodiff.py` for numeric and structural checks.
