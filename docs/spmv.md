# SpMV Design Notes

## 1) Scope

This document is the long-lived SpMV design entry for this repo. It currently covers:
- TCSR and CSR lowering paths;
- backend reference kernels and their storage location;
- tile-local ping-pong vector strategy;
- performance modeling terms that dominate one-iteration latency.

## 2) Implemented SpMV Paths

### 2.1 Components

- `src/backend/tcsr.py`
  - tiled sparse representation and preprocessing.
- `src/backend/csr.py`
  - standard CSR representation and preprocessing.
- `src/backend/sparse_to_aie.py`
  - `SparseToAIEPass` supports both `tcsr` and `csr`.
  - `tcsr` lowering injects `@spmv_tcsr` call and `@tcsr_*` globals.
  - `csr` lowering injects `@spmv_csr` call and `@csr_*` globals.
- `src/backend/aie_spmv_kernel.cpp`
  - TCSR-oriented reference kernel.
- `src/backend/aie_csr_spmv_kernel.cpp`
  - CSR-oriented reference kernel.

### 2.2 Kernel Placement Recommendation

Current kernels are under `src/backend/`.

Recommended future layout:
- `src/backend/kernels/spmv/tcsr/`
- `src/backend/kernels/spmv/csr/`
- `src/backend/kernels/solver/*/` for solver-fused kernels

This keeps pure format-level SpMV kernels separate from solver-specific fused kernels.

## 3) AIE Tile Memory Strategy

For iterative SpMV workloads:
- keep current vector in tile local `RAM A`;
- stream sparse matrix payload from DDR;
- write next vector into tile local `RAM B`;
- swap A/B on next iteration (ping-pong).

Benefits:
- minimizes repeated external vector fetches;
- enables overlap between compute and local-buffer handoff;
- maps naturally to fixed-matrix iterative solvers.

## 4) Performance Modeling

### 4.1 Why these terms matter

One-iteration SpMV performance is heavily bounded by:
- tile stream bandwidth (bits/cycle);
- CSR element width (`value_bit_width + index_bit_width`);
- vector local-read parallelism (`vector_reads_per_cycle`).

These directly determine whether the iteration is stream-bound, vector-read-bound, or compute-bound.

### 4.2 Model used in code

Implemented in `src/backend/spmv_performance_model.py`:

- Matrix stream bits:
  - `nnz * (value_bit_width + index_bit_width) + (num_rows + 1) * index_bit_width`
- Matrix stream cycles:
  - `matrix_bits / tile_stream_bandwidth_bits_per_cycle`
- Vector read cycles:
  - `nnz / vector_reads_per_cycle`
- Vector write cycles:
  - `num_rows / vector_writes_per_cycle` (optional, when set)
- Compute cycles:
  - `nnz / macs_per_cycle`
- Total cycles:
  - `max(matrix_stream_cycles, vector_read_cycles, vector_write_cycles, compute_cycles)`
- Time (microseconds):
  - `total_cycles / freq_mhz`

## 5) Gset Benchmark Modeling Test

Test file: `tests/test_spmv_performance_model.py`

What it does:
- parses one Gset matrix (currently `benchmarks/Gset/G1`);
- evaluates one-iteration time under multiple architecture configs;
- checks monotonic behavior (`narrow > balanced > wide`) and prints estimated time.

Run:

```bash
python -m pytest tests/test_spmv_performance_model.py -q -s
```

## 6) Notes

- Gset input is parsed as edge list; for SpMV modeling we treat it as symmetric by default (`nnz = 2 * edges`).
- This is a throughput model for design-space exploration, not a cycle-accurate hardware simulator.
