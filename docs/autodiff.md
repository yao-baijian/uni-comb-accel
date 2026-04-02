# Autodiff Frontend (JAX -> MLIR)

## Overview

`src/compiler/autodiff.py` provides a frontend bridge from user Python energy
functions to MLIR by tracing JAXPR for both forward and backward computations.

Pipeline:

1. `jax.make_jaxpr(func)(*example_args)` for forward graph.
2. `jax.grad(func, argnums=... )` + `make_jaxpr` for backward graph.
3. `JaxprToMLIR` lowers each JAXPR into a standalone MLIR `func.func`.
4. `combine_modules` merges forward/backward into one module.

## OMEinsum Contraction Order Optimization

The frontend now supports optional contraction-order optimization through
Julia packages `OMEinsum.jl` and `OMEinsumContractionOrders.jl`.

Install (in Julia):

```julia
using Pkg
Pkg.add("OMEinsum")
Pkg.add("OMEinsumContractionOrders")
Pkg.add("JSON")
```

Python side behavior:

- `JaxprToMLIR` detects a pure 2D matrix-chain (`A @ B @ C @ ...`) in forward JAXPR.
- It calls Julia to get pairwise contraction order.
- If Julia or packages are unavailable, it falls back to Python DP order.
- MLIR emits `linalg.matmul` in the optimized pairwise sequence.

Current scope: 2D matrix-chain contractions. General einsum hypergraph
optimization is structured for extension but not fully lowered yet.

## Supported Primitives

- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Linear algebra: `dot_general` (lowered to `linalg.matmul`), `broadcast_in_dim`, `transpose`
- Math: `exp`, `log`, `sin`, `cos`
- Reduction: `reduce_sum` (`linalg.reduce` skeleton), `reduce_max` placeholder
- Utility: `reshape`, `squeeze`, `convert_element_type`, `copy`

Unsupported primitives raise `NotImplementedError` with the primitive name.

## Intermediate Storage Strategy

The converter emits explicit temporary `memref.alloc` / `memref.dealloc` for
non-scalar intermediates to model the lifetime of forward-pass values that may
be reused in backward lowering.

Current implementation is a pragmatic frontend scaffold and can be extended to
materialize real save-for-backward buffers and pass them as explicit arguments.

## API Integration

`src/api.py` exposes:

```python
compile_energy_function(func, example_args, target="aie", output_dir="build")
```

OMEinsum knobs:

```python
compile_energy_function(
	func,
	example_args,
	target="aie",
	output_dir="build",
	use_omeinsum=True,
	julia_cmd="julia",
)
```

It generates:

- `<name>.forward.mlir`
- `<name>.backward.mlir`
- `<name>.combined.mlir`
- `<name>.combined.opt.mlir`

Then invokes ARIES backend for optimization/codegen.

## Extending Primitive Support

Add new primitive lowerings in `JaxprToMLIR._lower_eqn`:

1. Match primitive name.
2. Add lowering helper producing MLIR ops.
3. Map output vars into `_var_map`.
4. Add tests in `benchmarks/test_autodiff.py` for numerical + structural checks.
