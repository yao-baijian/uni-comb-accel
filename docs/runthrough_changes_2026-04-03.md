# Runthrough Changes (2026-04-03)

This file records the concrete changes made during the recent bring-up and build-validation session.

## 1. Python/Compile Path Changes

- Added shape-policy compile controls in `src/api.py`:
  - `shape_policy`: `exact | bucket | max_shape`
  - `shape_buckets`
  - `max_shape`
- Added compile-shape helpers in `src/api.py`:
  - `_select_compile_shapes`
  - `_shape_fits`
  - `_validate_shape_within`
  - `_build_trace_args_for_shapes`
- Added compile metadata to artifacts:
  - `shape_policy`
  - `actual_input_shapes`
  - `compile_input_shapes`
- Fixed legacy frontend return-dtype mismatch (`f32 -> f64` cast issue) by aligning `PythonToMLIR(return_dtype=precision.mlir_dtype)` in `compile_energy_function_legacy`.

## 2. Problem Session Cache Changes

- Updated signature inputs in `src/runtime/problem_session.py` to include:
  - `shape_policy`
  - `shape_buckets`
  - `max_shape`
- Fixed cache reuse false negatives:
  - introduced `_looks_like_path(...)`
  - manifest validation now checks existence only for path-like artifact values, not metadata strings.

## 3. Test Additions

- Added `tests/compile_test.py` with:
  - compile-flow smoke test (mock ARIES backend)
  - shape-policy cache hit/recompile behavior test
- Verified new tests pass in selected environment (`2 passed`).

## 4. ARIES Build/Tooling Bring-Up

- Rebuilt ARIES toolchain and confirmed binaries:
  - `tools/ARIES/build/bin/aries-opt`
  - `tools/ARIES/build/bin/aries-translate`
- Resolved CMake generator mismatch during build by clearing stale ARIES build cache.

## 5. Makefile Local-Environment Migration

Updated Makefiles to prefer local installed Vitis paths (instead of hardcoded 2023.2/lab paths):

- `tools/ARIES/templates/Makefile_VCK190`
- `tools/ARIES/example_new/example_Versal/example_gemm/example_project_small/project/Makefile`
- `tools/ARIES/example_new/example_Versal/example_gemm/example_project_large/project/Makefile`
- `tools/ARIES/example_new/example_Versal/example_mttkrp/example_project_small/project/Makefile`
- `tools/ARIES/example_new/example_Versal/example_ttmc/example_project_small/project/Makefile`
- `tools/ARIES/example_new/example_NPU/example_gemm/example_gemm_i16/project/Makefile`

Key migration changes:

- `XILINX_VITIS ?= /tools/Xilinx/2025.1/Vitis`
- platform auto-discovery from `${XILINX_VITIS}/base_platforms`
- dynamic `PLATFORM_TAG` and derived temp/build dirs
- `SYSROOT_PATH` and `EDGE_COMMON_SW` changed to override-friendly defaults
- NPU Makefile now uses local `ARIES_ROOT` and local include paths (removed hardcoded `/home/arclab/...`).

## 6. Real Build Validation Outcome

- Real non-mock compile from this repository succeeded to IR/C++ outputs in `build/real_compile_manual/`.
- Hardware image generation (`.xsa/.xclbin`) is still blocked on this machine due to missing target device parts in Vitis/Vivado installs:
  - examples failed on missing `xcvc1902` / `xcvm1802` device part support.

## 7. Documentation Restructuring

- Simplified `README.md` to keep only:
  - prerequisites
  - installation commands (project + subprojects)
  - latest progress
  - one simple end-to-end example
  - top-level and first-level architecture
- Moved detailed architecture descriptions out of README into:
  - `docs/architecture_overview.md`
- Merged AIE docs and normalized naming:
  - kept `docs/aie_model.md`
  - removed `docs/AIE_MODEL_IMPLEMENTATION.md`
- Unified documentation style by normalizing `docs/autodiff.md` section structure.
