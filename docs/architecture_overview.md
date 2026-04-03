# Architecture Overview

This document keeps the first-level subsystem responsibilities that were removed from `README.md`.

## Source Architecture (`src/`)

- `src/api.py`
  - Top-level compile/define/solve APIs.
  - Connects frontend, backend, runtime, precision, and shape-policy options.

- `src/compiler/`
  - `frontend.py`: manual Python AST -> MLIR lowering.
  - `autodiff.py`: JAXPR-based forward/backward MLIR generation.
  - `aie_sparse_dialect.py`: sparse op semantics scaffold.

- `src/backend/`
  - `aries_backend.py`: invokes `aries-opt` and `aries-translate`.
  - `sparse_to_aie.py`: sparse rewrite pass to AIE-friendly form.
  - `tcsr.py`: sparse matrix conversion to TCSR.
  - `precision.py`, `sparse_formats.py`: interface normalization.

- `src/runtime/`
  - `problem_session.py`: define/solve two-step flow, artifact cache, signature logic.
  - Board runtime abstractions and load-state checks.

- `src/testing/`
  - AIE performance benchmark harness and presets.

- `src/sbm/`
  - SBM/QUBO related helper conversions.

## External Subprojects (`tools/`)

- `tools/ARIES/`
  - Main compiler backend/toolchain used by this project.

- `tools/FEM/`
  - FEM-side references/benchmarks used for comparisons.

## Test Layout

- `tests/test_autodiff.py`: autodiff frontend smoke + gradient checks.
- `tests/test_spmv.py`: sparse/TCSR path tests.
- `tests/test_fem_autodiff.py`: FEM-style objective parity checks.
- `tests/compile_test.py`: compile pipeline smoke + shape-policy cache behavior.

## Build Artifacts

- `build/`: generated MLIR/code artifacts and local cache outputs.
- `build/problem_cache/`: cached problem definitions and compile outputs.
