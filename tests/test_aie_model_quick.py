#!/usr/bin/env python
"""Quick verification that AIE model and benchmark work correctly."""

import sys
sys.path.insert(0, '.')

# Test 1: Quick estimation
from src.aie_perf_model import estimate_aie_time
time_ms = estimate_aie_time(1024, 4096, 0.5, 32, 32, 32)
print(f"Example 1 - Quick Estimation:")
print(f"  1024-var MaxCut: {time_ms:.3f} ms\n")

# Test 2: Standard benchmark suite
from src.testing.aie_benchmark import create_standard_suite
bench = create_standard_suite()
results = bench.run_all(tile_rows=32, tile_cols=32, precision_bits=32)
print("Example 3 - Standard Benchmark Suite:")
for r in results:
    print(f"  {r.problem_name:20s}: {r.predicted_time_ms:7.3f} ms")
print()

# Test 3: Sparsity impact
from src.testing.aie_benchmark import create_sparse_suite
bench = create_sparse_suite()
results = bench.run_all()
print("Example 4 - Sparsity Impact Analysis:")
for r in results:
    print(f"  {r.problem_name:25s} {r.sparsity:9.1%} -> {r.predicted_time_ms:9.3f} ms")
print()

# Test 4: Precision impact
from src.aie_perf_model import AIEPerformanceModel
model = AIEPerformanceModel()
print("Example 5 - Precision Impact (1024-var problem):")
for bits in [32, 16, 8, 4]:
    cost = model.estimate_cost(
        num_variables=1024,
        num_clauses=4096,
        sparsity=0.5,
        precision_bits=bits
    )
    print(f"  {bits:2d}-bit: {cost.total_ms():8.3f} ms")
print()

# Test 5: Cost breakdown
print("Example 8 - Detailed Cost Breakdown:")
model = AIEPerformanceModel(verbose=False)
cost = model.estimate_cost(
    num_variables=2048,
    num_clauses=8192,
    sparsity=0.5,
    precision_bits=32,
)
print(f"  Initialization:  {cost.initialization_ms:.3f} ms")
print(f"  Compute:         {cost.compute_ms:.3f} ms")
print(f"  Memory Access:   {cost.memory_access_ms:.3f} ms")
print(f"  Communication:   {cost.communication_ms:.3f} ms")
print(f"  Finalization:    {cost.finalization_ms:.3f} ms")
print(f"  TOTAL:           {cost.total_ms():.3f} ms")
print()

print("✓ All examples completed successfully!")
