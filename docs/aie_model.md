# AIE Execution-Time Performance Model

This document describes the AIE performance model for predicting execution time of compiled QUBO/Ising problems on Xilinx AIE (Adaptive Intelligent Engine) FPGAs.

## Overview

The AIE Performance Model estimates execution time by decomposing computation into five phases:

1. **Initialization** - Load problem data, initialize state, setup tiles
2. **Compute** - Core QUBO/Ising computation with sparse/dense operations
3. **Memory Access** - Account for L1/L2 cache misses and DRAM latencies
4. **Communication** - Inter-tile routing overhead and synchronization
5. **Finalization** - Output aggregation and result write-back

## Architecture Parameters

The model requires FPGA architecture specifications:

```python
from src.aie_perf_model import AIEArchitecture

arch = AIEArchitecture(
    num_tiles=16,                    # Total AIE tiles available
    dram_size_mb=128,                # Total DRAM capacity (MB)
    dram_bandwidth_gb_s=16.0,        # Peak DRAM bandwidth (GB/s)
    pe_per_tile=4,                   # Processing elements per tile
    compute_units_per_pe=4,          # ALUs/multipliers per PE
    l1_cache_kb=32,                  # Per-tile L1 cache (KB)
    l2_bandwidth_gb_s=32.0,          # Inter-tile L2 bandwidth (GB/s)
    tile_frequency_ghz=1.0           # Tile operating frequency (GHz)
)
```

### Default Values

The default `AIEArchitecture()` represents a typical mid-range AIE configuration:
- 16 tiles in a 4×4 mesh
- 128 MB of DRAM with 16 GB/s bandwidth
- 4 PEs per tile, 4 compute units per PE (16 total compute units per tile)
- 32 KB L1 cache per tile

## Cost Breakdown

### Phase 1: Initialization (`initialization_ms`)

```
Time = DRAM_Latency + Data_Transfer + Tile_Setup

Data_Transfer = (num_variables × precision_bytes + num_clauses × 2 × precision_bytes) / dram_bandwidth_gb_s
Tile_Setup = 0.1 × num_tiles  (synchronization overhead)
```

For a 1024-variable problem with 4K clauses and FP32:
- Data = (1024 + 4096×2) × 4 bytes ≈ 36 KB
- Transfer ≈ 36 KB / 16 GB/s ≈ 0.002 ms
- Tile setup ≈ 0.1 × 16 = 1.6 ms
- **Total ≈ 2 ms**

### Phase 2: Compute (`compute_ms`)

Core computation cost depends on:
- **Effective clauses** = num_clauses × sparsity
- **Active tiles** = min(num_tiles, ⌈effective_clauses / (tile_rows × tile_cols)⌉)
- **Peak FLOPs** = active_tiles × pe_per_tile × compute_units_per_pe
- **Total FLOPs** = effective_clauses × ops_per_clause × tree_reduction_factor

```
Cycles = Total_FLOPs / (Peak_FLOPs × precision_factor)
Time = Cycles / (tile_frequency × 1000)  + overhead × 1.15
```

**Precision factor:**
- FP32: 1.0
- FP16: 0.8 (faster, less saturating compute)
- INT8: 0.65 (good packing, but more conversions)
- INT4: 0.5 (minimal storage, reduced precision overhead)

For a 1024-var, 50% sparse problem on 16 tiles:
- Effective clauses ≈ 2048
- Active tiles ≈ min(16, 2) = 2  
- Peak FLOPs ≈ 2 × 4 × 4 = 32 per cycle
- Total FLOPs ≈ 2048 × 2 × 10 (tree reduction) ≈ 41K
- Cycles ≈ 41K / 32 ≈ 1280
- Time ≈ 1280 / 1000 ≈ 1.3 ms × 1.15 ≈ **1.5 ms**

### Phase 3: Memory Access (`memory_access_ms`)

Cache miss penalty depends on working-set size:

```
if working_set ≤ L1_cache:
    miss_rate = 5%
elif working_set ≤ DRAM_size:
    miss_rate = working_set / (L1_cache × 1.5)  (capped at 30%)
else:
    miss_rate = 80%  (thrashing)

Stall_Cycles = num_memory_ops × miss_rate × 10  (10-cycle miss penalty)
Time = Stall_Cycles / (tile_frequency × 1000)
```

For 1024-variable problem:
- Working set ≈ 1024 × 4 bytes = 4 KB (fits in L1)
- Miss rate ≈ 5%
- Memory ops ≈ 2048 × 2 = 4096
- Stall cycles ≈ 4096 × 0.05 × 10 ≈ 2048
- Time ≈ 2048 / 1000 ≈ **2 ms**

### Phase 4: Communication (`communication_ms`)

Inter-tile communication and synchronization:

```
Boundary_Data ≈ effective_clauses × header_bytes × avg_hops
Stream_Time = Boundary_Data / l2_bandwidth_gb_s
Sync_Time = num_barriers × 0.05  (typically 3 barriers)
```

For typical 4×4 mesh:
- Avg hops ≈ 4
- Boundary data ≈ 2048 × 2 × 4 ≈ 16 KB
- Stream time ≈ 16 KB / 32 GB/s ≈ 0.5 ms
- Sync time ≈ 3 × 0.05 = 0.15 ms
- **Total ≈ 0.65 ms**

### Phase 5: Finalization (`finalization_ms`)

Result aggregation and write-back:

```
Result_Size = num_variables × precision_bytes + 64
Writeout_Time = Result_Size / dram_bandwidth_gb_s
Cleanup = 0.05 ms
```

For 1024 variables with FP32:
- Result size ≈ 4 KB + 64 bytes ≈ 4 KB
- Writeout ≈ 4 KB / 16 GB/s ≈ 0.25 ms
- **Total ≈ 0.3 ms**

## Total Estimated Time

```
Total = Init + Compute + Memory + Communication + Finalize
Example: 2 + 1.5 + 2 + 0.65 + 0.3 ≈ 6.5 ms
```

## Usage Examples

### Quick Estimation

```python
from src.aie_perf_model import estimate_aie_time

# Estimate time for a 1024-variable problem
time_ms = estimate_aie_time(
    num_variables=1024,
    num_clauses=4096,
    sparsity=0.5,
    precision_bits=32
)
print(f"Estimated: {time_ms:.2f} ms")
```

### Model with Custom Architecture

```python
from src.aie_perf_model import AIEArchitecture, AIEPerformanceModel

# High-end AIE (32 tiles, more bandwidth)
arch = AIEArchitecture(
    num_tiles=32,
    dram_bandwidth_gb_s=32.0,
    tile_frequency_ghz=1.5
)

model = AIEPerformanceModel(architecture=arch)
cost = model.estimate_cost(
    num_variables=2048,
    num_clauses=8192,
    sparsity=0.6,
    tile_rows=32,
    tile_cols=32,
    precision_bits=16
)

print(f"Breakdown:")
print(f"  Initialization: {cost.initialization_ms:.3f} ms")
print(f"  Compute: {cost.compute_ms:.3f} ms")
print(f"  Memory: {cost.memory_access_ms:.3f} ms")
print(f"  Communication: {cost.communication_ms:.3f} ms")
print(f"  Finalization: {cost.finalization_ms:.3f} ms")
print(f"  Total: {cost.total_ms():.3f} ms")
```

### Benchmark Suite

```python
from src.testing.aie_benchmark import create_standard_suite

# Create and run standard benchmark
bench = create_standard_suite()
results = bench.run_all(
    tile_rows=32,
    tile_cols=32,
    precision_bits=32
)

# Save results
filepath = bench.save_results()
print(bench.report())
```

## Tuning Guidelines

### Choosing Tile Sizes

Tile sizes (`tile_rows`, `tile_cols`) affect:
- **Smaller tiles** (16×16): Lower latency, more tile boundaries, higher communication overhead
- **Larger tiles** (64×64): Higher latency, fewer tile boundaries, lower overhead

Heuristic from `_build_aie_config()`:
```
if estimated_intermediates ≤ 262 KB: tile_size = 16
if estimated_intermediates ≤ 1 MB:   tile_size = 32
if estimated_intermediates ≤ 4 MB:   tile_size = 64
else:                                tile_size = 128
```

### Precision Trade-offs

| Precision | Storage | FLOPs Speedup | Compute Cost Factor | Use Case |
|-----------|---------|---------------|---------------------|----------|
| FP32 | 4 B/num | 1× | 1.0 | Baseline, highest accuracy |
| FP16 | 2 B/num | 2× | 0.8 | Good balance, reduced precision |
| INT8 | 1 B/num | 4× | 0.65 | Lower precision, faster |
| INT4 | 0.5 B/num | 8× | 0.5 | Minimal footprint, quantization |

For 1K problem: FP16 saves ~25% time vs FP32; INT8 saves ~50%.

### Sparsity Impact

Model accounts for sparsity linearly:
- **Dense (sparsity=1.0)**: All interactions computed → longest time
- **Sparse (sparsity=0.1)**: 10% interactions → ~10% compute time
- **Very sparse (sparsity=0.01)**: Dominated by overhead → small further gains

### Cache Blocking

To minimize memory stalls:
- Keep working set ≤ L1 cache size (32 KB typical)
- Process problems in batches to amortize initialization
- Use tiling to improve L1 hit rate

## Integration with FPGA Execution

### Model-Only (Fast Iteration)

```python
bench = AIEBenchmark(use_model=True, use_fpga=False)
# Add presets and run_all()
# Results contain predicted_time_ms only
```

### Model + FPGA Comparison

```python
bench = AIEBenchmark(use_model=True, use_fpga=True)
# Requires compiled AIE code and board access
# Results contain both predicted_time_ms and actual_time_ms
# accuracy_error_pct() measures |predicted - actual| / actual
```

### FPGA Execution Only (Requires Integration)

```
bench = AIEBenchmark(use_model=False, use_fpga=True)
# Requires full FPGA harness (not yet implemented)
# Would call api.define_problem() + BoardRuntime.load() + solve_problem()
# Measures wall-clock execution time
```

## Model Limitations & Assumptions

1. **Linear sparsity scaling**: Assumes computation time scales linearly with interaction count. May underestimate overhead for very sparse problems.

2. **Uniform tile utilization**: Assumes clauses are evenly distributed across tiles. Real imbalances may increase actual time by 5-15%.

3. **No DRAM contention**: Assumes DRAM bandwidth is fully available. Shared access may reduce effective bandwidth.

4. **No pipeline stalls**: Does not model data dependencies or instruction-level parallelism beyond 15% overhead factor.

5. **Precision scaling**: Assumes FP16/INT8/INT4 provide listed speedups. Actual gains depend on compiler code generation.

6. **Cache hierarchy simplified**: Uses simplified miss-rate model; actual miss patterns depend on data layout and reuse distance.

## Future Improvements

- [ ] Add instruction-level pipeline model
- [ ] Integrate with actual AIE compiler for more accurate operation counts
- [ ] Support variable-size tile configurations per problem
- [ ] Add data reuse & tiling optimization analysis
- [ ] Validate against real FPGA measurements
- [ ] Support batch/streaming workloads
- [ ] Add memory traffic simulation for multi-problem scenarios

## References

- Xilinx AIE Documentation: https://github.com/arc-research-lab/Aries
- Performance Modeling Techniques: https://people.eecs.berkeley.edu/~dimakis/papers/roofline.pdf
- Cache Simulation Models: https://dlbeer.co.uk/files/papers/pisa.pdf
