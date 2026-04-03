"""Example: AIE performance benchmark suite demonstration.

This script shows how to:
1. Create benchmark problems
2. Run predictions with the AIE performance model
3. Compare sparsity, precision, and problem-size impacts
4. Save and report results
"""

from src.aie_perf_model import AIEArchitecture, estimate_aie_time
from src.testing.aie_benchmark import (
    AIEBenchmark,
    ProblemPreset,
    create_standard_suite,
    create_sparse_suite,
    create_precision_suite,
)


def example_quick_estimation():
    """Quick single-problem estimation."""
    print("=" * 60)
    print("Example 1: Quick Estimation")
    print("=" * 60)

    time_ms = estimate_aie_time(
        num_variables=1024,
        num_clauses=4096,
        sparsity=0.5,
        precision_bits=32,
    )
    print(f"Estimated time for 1024-var MaxCut: {time_ms:.3f} ms")
    print()


def example_custom_architecture():
    """Estimate with custom FPGA architecture."""
    print("=" * 60)
    print("Example 2: Custom FPGA Architecture")
    print("=" * 60)

    # Optimized high-end AIE configuration
    arch = AIEArchitecture(
        num_tiles=32,                    # More tiles
        dram_size_mb=512,                # More DRAM
        dram_bandwidth_gb_s=32.0,        # Higher bandwidth
        pe_per_tile=8,                   # More PEs per tile
        compute_units_per_pe=4,
        tile_frequency_ghz=1.5,          # Higher frequency
    )

    # Compare standard vs optimized
    from src.aie_perf_model import AIEPerformanceModel

    model_std = AIEPerformanceModel()
    model_opt = AIEPerformanceModel(architecture=arch)

    cost_std = model_std.estimate_cost(
        num_variables=2048,
        num_clauses=8192,
        sparsity=0.5,
        precision_bits=32,
    )

    cost_opt = model_opt.estimate_cost(
        num_variables=2048,
        num_clauses=8192,
        sparsity=0.5,
        precision_bits=32,
    )

    print(f"2048-variable QUBO:")
    print(f"  Standard AIE:  {cost_std.total_ms():.3f} ms")
    print(f"  Optimized AIE: {cost_opt.total_ms():.3f} ms")
    print(f"  Speedup:       {cost_std.total_ms() / cost_opt.total_ms():.2f}×")
    print()


def example_standard_benchmark():
    """Run standard benchmark suite."""
    print("=" * 60)
    print("Example 3: Standard Benchmark Suite")
    print("=" * 60)

    bench = create_standard_suite()
    results = bench.run_all(tile_rows=32, tile_cols=32, precision_bits=32)

    for result in results:
        print(
            f"{result.problem_name:20s}: "
            f"{result.num_variables:5d} vars, "
            f"predicted: {result.predicted_time_ms:7.3f} ms"
        )
    print()


def example_sparsity_impact():
    """Analyze sparsity vs execution time."""
    print("=" * 60)
    print("Example 4: Sparsity Impact Analysis")
    print("=" * 60)

    bench = create_sparse_suite()
    results = bench.run_all(tile_rows=32, tile_cols=32, precision_bits=32)

    print(f"{'Problem':25s} {'Sparsity':10s} {'Time (ms)':10s}")
    print("-" * 50)
    for result in results:
        print(
            f"{result.problem_name:25s} "
            f"{result.sparsity:9.1%} "
            f"{result.predicted_time_ms:9.3f} ms"
        )
    print()


def example_precision_impact():
    """Analyze precision vs execution time."""
    print("=" * 60)
    print("Example 5: Precision Impact Analysis")
    print("=" * 60)

    bench = AIEBenchmark(use_model=True)

    presets = [
        ProblemPreset(
            name="fp32", num_variables=1024, num_clauses=4096, sparsity=0.5
        ),
        ProblemPreset(
            name="fp16", num_variables=1024, num_clauses=4096, sparsity=0.5
        ),
        ProblemPreset(
            name="int8", num_variables=1024, num_clauses=4096, sparsity=0.5
        ),
        ProblemPreset(
            name="int4", num_variables=1024, num_clauses=4096, sparsity=0.5
        ),
    ]

    for preset in presets:
        bench.add_preset(preset)

    precision_bits_map = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}

    print(f"{'Precision':10s} {'Time (ms)':10s} {'vs FP32':10s}")
    print("-" * 35)

    results = []
    for name, prec_bits in precision_bits_map.items():
        result = bench.run(name, tile_rows=32, tile_cols=32, precision_bits=prec_bits)
        results.append(result)

    fp32_time = results[0].predicted_time_ms
    for result in results:
        speedup = fp32_time / result.predicted_time_ms if result.predicted_time_ms else 1
        print(
            f"{result.precision_bits:3d}-bit    "
            f"{result.predicted_time_ms:9.3f} ms "
            f"{speedup:8.2f}×"
        )
    print()


def example_custom_preset():
    """Create custom problem preset and benchmark."""
    print("=" * 60)
    print("Example 6: Custom Problem Preset")
    print("=" * 60)

    bench = AIEBenchmark(use_model=True, output_dir="./benchmark_output")

    # Define custom graph coloring problem
    coloring_preset = ProblemPreset(
        name="graph_coloring_8k",
        num_variables=8192,
        num_clauses=32768,
        sparsity=0.75,
        description="8K-node chromatic scheduling on regular 4-connectivity graph",
        metadata={
            "problem_type": "graph_coloring",
            "graph_connectivity": 4,
            "colors": 4,
        },
    )

    bench.add_preset(coloring_preset)

    print(f"Running: {coloring_preset.description}")
    result = bench.run(
        "graph_coloring_8k",
        tile_rows=64,
        tile_cols=64,
        precision_bits=16,  # Use FP16 for faster execution
    )

    print(f"  Predicted time: {result.predicted_time_ms:.3f} ms")
    print(f"  Configuration: {result.tile_rows}×{result.tile_cols} tiles, {result.precision_bits}-bit")
    print()

    # Save for later analysis
    filepath = bench.save_results("coloring_benchmark.json")
    print(f"Results saved to: {filepath}")
    print()


def example_batch_estimation():
    """Estimate multiple problems in batch."""
    print("=" * 60)
    print("Example 7: Batch Estimation")
    print("=" * 60)

    from src.aie_perf_model import AIEPerformanceModel

    model = AIEPerformanceModel(verbose=False)

    problems = [
        {
            "name": "MaxCut-1K",
            "num_variables": 1024,
            "num_clauses": 4096,
            "sparsity": 0.5,
        },
        {
            "name": "MaxCut-2K",
            "num_variables": 2048,
            "num_clauses": 8192,
            "sparsity": 0.5,
        },
        {
            "name": "MaxCut-4K",
            "num_variables": 4096,
            "num_clauses": 16384,
            "sparsity": 0.5,
        },
    ]

    batch_result = model.estimate_batch_cost(problems)

    print("Batch Estimation Results:")
    print(f"  Total problems: {batch_result['num_problems']}")
    print(f"  Total time: {batch_result['total_ms']:.3f} ms")
    print(f"  Average per problem: {batch_result['average_ms']:.3f} ms")
    print()


def example_cost_breakdown():
    """Show detailed cost breakdown."""
    print("=" * 60)
    print("Example 8: Detailed Cost Breakdown")
    print("=" * 60)

    from src.aie_perf_model import AIEPerformanceModel

    model = AIEPerformanceModel(verbose=True)

    print("Estimating 2048-variable QUBO with 50% sparsity...\n")
    cost = model.estimate_cost(
        num_variables=2048,
        num_clauses=8192,
        sparsity=0.5,
        tile_rows=32,
        tile_cols=32,
        precision_bits=32,
    )

    print("\n" + "=" * 40)
    print("Cost Breakdown:")
    print("=" * 40)
    print(f"  Initialization:  {cost.initialization_ms:.3f} ms")
    print(f"  Compute:         {cost.compute_ms:.3f} ms")
    print(f"  Memory Access:   {cost.memory_access_ms:.3f} ms")
    print(f"  Communication:   {cost.communication_ms:.3f} ms")
    print(f"  Finalization:    {cost.finalization_ms:.3f} ms")
    print(f"  {'─' * 40}")
    print(f"  TOTAL:           {cost.total_ms():.3f} ms")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AIE PERFORMANCE MODEL EXAMPLES")
    print("=" * 60 + "\n")

    example_quick_estimation()
    example_custom_architecture()
    example_standard_benchmark()
    example_sparsity_impact()
    example_precision_impact()
    example_custom_preset()
    example_batch_estimation()
    example_cost_breakdown()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
