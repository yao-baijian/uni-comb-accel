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
- SBM path:
	- `src/sbm/problem_to_ising.py`: shared MaxCut, balanced min-cut, and QPLIB
		problem-to-Ising conversions for SBM-style solvers.
	- `src/backend/aie_sbm_kernel.cpp`: AIE-oriented SBM kernel helpers for
		SpMV and time-evolution simulation.
- Tests (`tests/`): pytest-discovered smoke tests for autodiff, FEM, and SpMV.

## Structure

- `src/` - source code
- `build/` - local build artifacts (ignored by git)
- `tests/` - pytest suites and smoke tests
- `tools/ARIES/` - ARIES toolkit (git submodule)
- `tools/FEM/` - FEM toolkit (git submodule)

## Setup

```bash
git submodule update --init --recursive
```

Run sparse test:

```bash
pytest tests/test_spmv.py -q
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
3. Add a test in `tests/test_autodiff.py` covering:
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

The energy-function compiler now accepts `gradient_mode="auto"|"manual"`,
`precision="fp32"|"fp16"|"int8"|"int4"`, and
`sparse_format="tcsr"|"csr"|"csr5"|"bcoo"|"original_csr"`. The precision
setting applies to the QUBO variables in SBM/FEM-style problems. `fp32`,
`fp16`, and `int8` are represented directly in the tracing path; `int4` is a
logical interface value carried in metadata with `int8` storage compatibility.
Only `tcsr` has a concrete sparse lowering path today; the other sparse formats
are carried as backend interface metadata for future implementation.

Run autodiff tests:

```bash
pytest tests/test_autodiff.py -q
```

## Define Then Solve (Two-Step Workflow)

The framework now treats problem use as two explicit phases:

1. `define_problem(...)`:
   - Builds a problem signature from:
	 - problem type
	 - energy function source
	 - target / gradient mode / sparse format / precision
	 - expected input variable shapes + auto AIE config mode
   - Reuses existing compiled AIE artifacts from cache when signature is unchanged.
   - Regenerates AIE code only when the signature changes (for example, problem type or compile options change).
	- Accepts optional `expected_input_shapes={"x": (...), "J": (...)}` to guide sizing.
	- If `expected_input_shapes` is omitted, defaults are inferred from `example_args`; missing dimensions fall back to `(1024,)`.
	- Auto-estimates intermediate variable footprint and emits an AIE config file (`*.aie_config.json`) that drives tile-size selection.

2. `solve_problem(...)`:
   - Uses a previously returned problem handle.
   - Checks whether matching code is loaded to the board runtime.
   - Can either error with a clear message or auto-load before solve.

Minimal example:

```python
from src.api import define_problem, solve_problem
from src.runtime import FileBoardRuntime

def energy(x, J):
	return (x @ J @ x).sum()

handle = define_problem(
	problem_type="sbm.maxcut",
	energy_function=energy,
	example_args=(x_example, j_example),
	expected_input_shapes={"x": (4096,), "J": (4096, 4096)},
	target="aie",
	gradient_mode="auto",
	sparse_format="tcsr",
	precision="fp16",
	auto_aie_config=True,
)

board = FileBoardRuntime(state_file="build/board_runtime/state.json")

result = solve_problem(
	handle,
	solver_fn=my_solver,
	board_runtime=board,
	auto_load_to_board=True,
)
```

This split avoids unnecessary AIE re-generation across repeated runs of the same
problem definition.

`FileBoardRuntime` persists loaded state across runs and also checks artifact
fingerprints, so a stale board load is treated as not-loaded and can be
re-loaded automatically.

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

## AIE Performance Model & Benchmarking

The framework includes a parametric performance model for predicting AIE execution time and a benchmarking framework to validate predictions.

### Performance Model

The AIE performance model (`src/aie_perf_model.py`) decomposes execution cost into five phases:

1. **Initialization**: Problem data load and tile setup
2. **Compute**: Core QUBO/Ising computation with sparse/dense patterns
3. **Memory Access**: Cache misses and DRAM latency stalls
4. **Communication**: Inter-tile routing and synchronization
5. **Finalization**: Result aggregation and write-back

Quick estimation:

```python
from src.aie_perf_model import estimate_aie_time

# Predict time for 1024-variable QUBO
time_ms = estimate_aie_time(
	num_variables=1024,
	num_clauses=4096,
	sparsity=0.5,
	precision_bits=32
)
print(f"Predicted: {time_ms:.2f} ms")
```

Custom architecture:

```python
from src.aie_perf_model import AIEArchitecture, AIEPerformanceModel

# High-end AIE configuration
arch = AIEArchitecture(
	num_tiles=32,
	dram_bandwidth_gb_s=32.0,
	tile_frequency_ghz=1.5,
)

model = AIEPerformanceModel(architecture=arch)
cost = model.estimate_cost(
	num_variables=2048,
	num_clauses=8192,
	sparsity=0.6,
	tile_rows=32,
	tile_cols=32,
	precision_bits=16,
)

print(f"Total: {cost.total_ms():.3f} ms")
print(f"  Compute: {cost.compute_ms:.3f} ms")
print(f"  Memory:  {cost.memory_access_ms:.3f} ms")
```

See `docs/aie_model.md` for detailed breakdown of cost calculations and tuning guidelines.

### Benchmarking Framework

The benchmark suite (`src/testing/aie_benchmark.py`) provides:

- **Problem presets**: Predefined QUBO/Ising problems at various scales
- **AIEBenchmark harness**: Runs problems and compares predicted vs actual times
- **Result tracking**: JSON export and analysis

Standard benchmark:

```python
from src.testing.aie_benchmark import create_standard_suite

bench = create_standard_suite()  # 1K, 2K, 4K variable problems
results = bench.run_all(
	tile_rows=32,
	tile_cols=32,
	precision_bits=32
)

# Save results
filepath = bench.save_results()
print(bench.report())
```

Specialized suites:

```python
from src.testing.aie_benchmark import (
	create_small_suite,      # < 1K variables
	create_sparse_suite,     # Varying sparsity levels
	create_precision_suite,  # Precision impact (FP32/FP16/INT8/INT4)
)
```

Custom benchmark:

```python
from src.testing.aie_benchmark import ProblemPreset, AIEBenchmark

bench = AIEBenchmark(use_model=True)
bench.add_preset(ProblemPreset(
	name="custom_problem",
	num_variables=8192,
	num_clauses=32768,
	sparsity=0.75,
	description="Custom graph problem"
))

result = bench.run("custom_problem")
print(f"Predicted time: {result.predicted_time_ms:.3f} ms")
```

Run example benchmarks:

```bash
python benchmarks/test_aie_model.py
```

This script demonstrates:
- Quick single-problem estimation
- Architecture comparison (standard vs optimized)
- Standard benchmark suite
- Sparsity impact analysis
- Precision trade-offs (FP32/FP16/INT8/INT4)
- Custom problem presets
- Batch estimation
- Detailed cost breakdown

### FPGA Integration (Future)

The benchmark framework supports optional FPGA execution for validation:

```python
bench = AIEBenchmark(use_model=True, use_fpga=True)
results = bench.run_all()

# Each result now has both predicted_time_ms and actual_time_ms
for result in results:
	error_pct = result.accuracy_error_pct()
	print(f"{result.problem_name}: {error_pct:.1f}% error")
```

FPGA execution requires integration with compiled AIE harness (not yet implemented).
