# Sparse Pipeline Change Log (Repo + ARIES)

Date: 2026-04-20
Scope: CSR/SpMV flow stabilization, SW emulation diagnostics, ARIES frontend/translation patching

## 1) Repository-side Important Changes

### 1.1 Backend sparse-lowering behavior (CSR support path)
- File: src/backend/csr.py
  - Added csr_from_payload(payload, dtype=np.float32) to construct CSRData directly from serialized metadata.
  - Validates keys (values, col_indices, row_ptr, shape) and shape consistency.
- File: src/backend/sparse_to_aie.py
  - Added direct CSR payload extraction from aries.csr_input.
  - CSR path now prefers direct payload -> csr_from_payload(...) before fallback matrix->generate_csr(...).
  - Keeps non-CSR path behavior for TCSR extraction/rewrite.
- File: src/backend/aries_backend.py
  - Fixed call-time sparse format override by constructing SparseToAIEPass with sparse_format passed to optimize(...).

### 1.2 Sparse lowering tests
- File: tests/test_sparse_lowering.py
  - Added regression: direct CSR payload is accepted and emits CSR lowering artifacts.
  - Added regression: optimize(..., sparse_format="csr") overrides backend default sparse format.

### 1.3 Notebook and SW emu flow (SpMV)
- File: tests/iter_spmv.ipynb
  - Uses row-block padded CSR with static MAX_NNZ_PER_ROW.
  - Defines explicit task_kernel + task_tile flow for spmv_rowblk.
  - Uses explicit aries.buffer declarations before aries.load.
  - Uses named xx = aries.arange(0, S_N) for load indexing.
  - Generates build artifact at build/iter_spmv_segmented.
- File: tests/test_sw_aie_mv_spmv.py
  - Real sw emu/aiesim test path for dense + spmv generated projects.
  - Includes conservative Makefile patching and known-failure xfail classifications.

## 2) ARIES-side Important Changes

### 2.1 Frontend kernel lowering for dynamic gather indices
- File: tools/ARIES/frontend/aries_ir_builder.py
- Class: KernelMLIRGenerator
- Key updates:
  - index_expr(...) now tracks affine/non-affine expression state.
  - Supports nested subscript index expressions (e.g., X[ColIdx[p]]) by materializing nested load.
  - Emits arith.index_cast for non-index scalar index values before memref.load indexing.
  - visit_Subscript / visit_Assign / visit_AugAssign switch between affine.* and memref.* ops based on affine status.

### 2.2 Translation kernel header emission robustness
- File: tools/ARIES/lib/Translation/EmitAriesCpp.cpp
- Function: emitKernelHeader(FuncOp func)
- Key updates:
  - Handles zero-result kernel FuncOp signatures safely.
  - Fallback rule: for zero explicit MLIR results, treat last memref argument as output_buffer.
  - Prevents malformed prototype generation (e.g., trailing comma / missing closure in adf_kernel.h).

### 2.3 Rebuild requirement after ARIES C++ translation changes
- Command used:
  - cmake --build tools/ARIES/build --target aries-translate -j8

## 3) Current Status (Known Remaining Blocker)

- Dense SW emu path: passing.
- SpMV SW emu path: currently xfail in tests.
- Current failure mode:
  - aiecompiler graph frontend crash while executing adf_graph.out.
  - Generated build/iter_spmv_segmented/project/aie/adf_graph.cpp is still incomplete (includes only, no graph object wiring).
  - adf_graph.h fallback stub is still being used in test harness.

### 3.1 Test harness nuance fixed (Makefile bypass persistence)
- File: tests/test_sw_aie_mv_spmv.py
- Problem:
  - Retry helper _patch_spmv_bypass_pass(...) rewrites top Makefile pass rule to cp $< $@.
  - That bypass persisted across later test runs and could hide true pass-stage behavior.
- Fix:
  - _patch_spmv_make_options(...) now restores normal pass rule (aries-opt + pipeline options) before first run.
  - First attempt always exercises real ARIES pass pipeline; bypass is only applied for explicit retry path.

## 4) Why This Matters for Other Sparse Formats

The same categories of failure are likely to reappear for COO/BCSR/blocked variants:
- Dynamic/non-affine gather indexing in kernels.
- Zero-result kernel signature emission mismatch in translation.
- Incomplete graph object emission/splitting in ADF C++ generation.

## 5) Recommended Patch Checklist for New Sparse Formats

When adding another sparse format, validate in this order:
1. Frontend lowering:
   - Any X[idx] where idx is computed from memory requires non-affine-safe lowering.
   - Confirm index_cast to MLIR index when needed.
2. Kernel signature emission:
   - Validate adf_kernel.h prototypes close correctly for zero-result and memref-out patterns.
3. Graph emission:
   - Confirm adf_graph.cpp includes graph class/object, port wiring, and main path when required.
4. End-to-end SW emu:
   - Run pytest -q tests/test_sw_aie_mv_spmv.py -rxX (or equivalent new-format test) and classify known failures explicitly.

## 6) Fast Reproduction Commands

From repository root:
- Re-generate sparse project via notebook cells in tests/iter_spmv.ipynb.
- Run SW emu tests:
  - pytest -q tests/test_sw_aie_mv_spmv.py -rxX
- Inspect latest failure log:
  - build/iter_spmv_segmented/project/run_sw_emu_pytest_first.log

## 7) Next ARIES Focus Area

Highest priority unresolved issue is ADF graph emission completeness for sparse flow:
- Compare generated aie/adf_graph.cpp against working dense/GEMM graph outputs.
- Patch ARIES translation path responsible for graph object/class emission and split output wiring for sparse kernel flows.
