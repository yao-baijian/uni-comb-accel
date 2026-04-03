# AIE Model

## Scope

This document merges the previous `AIE_MODEL_IMPLEMENTATION.md` and `aie_model.md`.
It covers both:

- the execution-time model design
- the current implementation and benchmark entry points

## Model Overview

The AIE performance model estimates runtime by splitting total cost into five phases:

1. Initialization
2. Compute
3. Memory access
4. Communication
5. Finalization

Primary implementation file:

- `src/aie_perf_model.py`

Quick API:

```python
from src.aie_perf_model import estimate_aie_time

ms = estimate_aie_time(
    num_variables=1024,
    num_clauses=4096,
    sparsity=0.5,
    precision_bits=32,
)
print(ms)
```

## Architecture Parameters

`AIEArchitecture` controls hardware assumptions used in estimation:

- tile count and frequency
- DRAM size and bandwidth
- per-tile compute resources
- cache-related capacity terms

Example:

```python
from src.aie_perf_model import AIEArchitecture, AIEPerformanceModel

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

print(cost.total_ms())
```

## Cost Terms

The model computes:

- `initialization_ms`
- `compute_ms`
- `memory_access_ms`
- `communication_ms`
- `finalization_ms`

And total:

- `total_ms()`

This gives a stable decomposition for tuning and regression tracking.

## Benchmark Framework

Benchmark harness files:

- `src/testing/aie_benchmark.py`
- `benchmarks/test_aie_model.py`

Capabilities:

- problem presets
- suite runs (`small`, `standard`, `sparse`, `precision`)
- JSON result export
- report generation

Example:

```python
from src.testing.aie_benchmark import create_standard_suite

bench = create_standard_suite()
results = bench.run_all(tile_rows=32, tile_cols=32, precision_bits=32)
print(bench.report())
```

## Current Status

Implemented and available:

- model API (`estimate_aie_time`, `AIEPerformanceModel`)
- benchmark harness and preset suites
- documentation and usage examples

Not yet fully closed-loop in this repository:

- automatic model-vs-board timing calibration
- full FPGA execution validation pipeline in one command

## Practical Notes

- The model is currently best used for relative comparison and tuning direction.
- Absolute numbers depend on board image, routing congestion, and installed device packages.
- For hardware runs in this repo, ensure Vitis/Vivado has required Versal device parts installed.
