# uni-comb-accel

Repository scaffold for combinatorial acceleration experiments.

## What Is Going On Now

This repository now has a first end-to-end path from Python energy functions to
ARIES-based AIE code generation, plus an initial sparse SpMV acceleration stack:

- Frontend (`src/compiler/frontend.py`): lowers a subset of Python AST into MLIR.
- Backend API (`src/api.py`): orchestrates lower -> optimize -> codegen.
- ARIES backend (`src/backend/aries_backend.py`): runs `aries-opt`/`aries-translate`
	and now performs sparse-op pre-lowering before ARIES passes.
- TCSR pipeline:
	- `src/backend/tcsr.py`: converts sparse matrices to TCSR and can emit C arrays.
	- `src/backend/sparse_to_aie.py`: rewrites sparse matmul patterns into
		`spmv_tcsr` kernel calls and injects TCSR metadata into MLIR module attrs.
	- `src/compiler/aie_sparse_dialect.py`: lightweight definition for
		`aie_sparse.spmv` op semantics.
	- `src/backend/aie_spmv_kernel.cpp`: AIE-oriented SpMV kernel implementation
		with tile traversal and double-buffered load/compute structure for simulation
		and downstream AIE specialization.
- Benchmark (`benchmarks/test_spmv.py`): validates TCSR SpMV correctness against
	NumPy/SciPy CPU reference and reports simple timing.

## Structure

- `src/` - source code
- `build/` - local build artifacts (ignored by git)
- `benchmarks/` - benchmark suites and scripts
- `tools/ARIES/` - ARIES toolkit (git submodule)
- `tools/FEM/` - FEM toolkit (git submodule)

## Setup

```bash
git submodule update --init --recursive
```

Run sparse benchmark:

```bash
pytest benchmarks/test_spmv.py -q
```

## Autodiff Frontend

### Design

`src/compiler/autodiff.py` implements a JAX-driven frontend:

1. Trace forward graph with `jax.make_jaxpr`.
2. Trace backward graph via `jax.grad` + `make_jaxpr`.
3. Lower JAXPR primitives into MLIR (`func`, `arith`, `math`, `linalg`,
   `tensor`, `memref`).
4. Merge forward/backward modules into one combined module for backend stages.

The high-level entrypoint is `compile_energy_function` in `src/api.py`.

### Supported JAX Primitives

- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Linear algebra: `dot_general`, `broadcast_in_dim`, `transpose`
- Math: `exp`, `log`, `sin`, `cos`
- Reduction: `reduce_sum`, `reduce_max` (scaffold)
- Utility: `reshape`, `squeeze`, `convert_element_type`, `copy`

Unsupported primitives raise `NotImplementedError` with clear primitive names.

### How To Extend

1. Add primitive dispatch in `JaxprToMLIR._lower_eqn`.
2. Implement lowering helper that emits MLIR for the op.
3. Add a benchmark test in `benchmarks/test_autodiff.py` covering:
   - JAX gradient numeric correctness
   - generated MLIR structural checks

### Usage Example

```python
import jax.numpy as jnp
from src.api import compile_energy_function

def energy(x):
	return jnp.sum(x * x)

artifacts = compile_energy_function(
	energy,
	example_args=(jnp.ones((16,), dtype=jnp.float32),),
	target="aie",
	output_dir="build",
)

print(artifacts)
```

Run autodiff benchmarks:

```bash
pytest benchmarks/test_autodiff.py -q
```

### OMEinsum (Julia) For Contraction Order

To optimize tensor contraction/matmul chain order from Python frontend, install
Julia packages:

```julia
using Pkg
Pkg.add("OMEinsum")
Pkg.add("OMEinsumContractionOrders")
Pkg.add("JSON")
```

Then call API with OMEinsum enabled (default enabled):

```python
compile_energy_function(
	func,
	example_args,
	use_omeinsum=True,
	julia_cmd="julia",
)
```

If Julia packages are unavailable, frontend falls back to Python dynamic
programming order for matrix-chain contractions.
