"""AIE performance benchmark framework for energy problem solving.

Provides:
1. Problem presets: standard test problems with known sizes/characteristics
2. AIEBenchmark: harness to run problems against model and/or actual FPGA
3. Result tracking and comparison (predicted vs actual time)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from src.aie_perf_model import AIEArchitecture, AIEPerformanceModel, OperationCost


@dataclass
class ProblemPreset:
    """Predefined QUBO/Ising problem for benchmarking.
    
    Attributes:
        name: Display name for the problem.
        num_variables: Number of Ising variables.
        num_clauses: Number of interactions.
        sparsity: Sparsity factor (0.0-1.0).
        description: Human-readable description.
        energy_fn: Optional callable that computes problem energy.
        example_args: Optional dict with example input variables for shape inference.
        metadata: Additional problem-specific metadata.
    """

    name: str
    num_variables: int
    num_clauses: int
    sparsity: float = 0.5
    description: str = ""
    energy_fn: Optional[Callable[[Any], float]] = None
    example_args: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize (excludes callables)."""
        d = asdict(self)
        d.pop("energy_fn", None)  # Exclude non-serializable callable
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ProblemPreset:
        """Deserialize from dict."""
        d_copy = d.copy()
        return ProblemPreset(**d_copy)


@dataclass
class BenchmarkResult:
    """Result of running a single problem benchmark."""

    problem_name: str
    num_variables: int
    num_clauses: int
    sparsity: float
    predicted_time_ms: Optional[float] = None
    actual_time_ms: Optional[float] = None
    energy_value: Optional[float] = None
    error_msg: Optional[str] = None
    tile_rows: int = 32
    tile_cols: int = 32
    precision_bits: int = 32
    timestamp: float = field(default_factory=time.time)

    def accuracy_error_pct(self) -> Optional[float]:
        """Compute |predicted - actual| / actual as percentage."""
        if self.predicted_time_ms is None or self.actual_time_ms is None:
            return None
        if self.actual_time_ms == 0:
            return None
        return (
            abs(self.predicted_time_ms - self.actual_time_ms)
            / self.actual_time_ms
            * 100.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> BenchmarkResult:
        """Deserialize from dict."""
        return BenchmarkResult(**d)


class AIEBenchmark:
    """Harness for AIE performance benchmarking.
    
    Supports running problems against:
    - AIE model: fast, deterministic cost prediction
    - Actual FPGA: requires compiledAIE code and board access
    
    Example:
        >>> benchmark = AIEBenchmark(use_model=True)
        >>> benchmark.add_preset(
        ...     ProblemPreset(
        ...         name="small_maxcut",
        ...         num_variables=256,
        ...         num_clauses=512,
        ...         description="Max-Cut on 256-node graph"
        ...     )
        ... )
        >>> result = benchmark.run("small_maxcut")
        >>> print(f"Predicted: {result.predicted_time_ms:.2f} ms")
    """

    def __init__(
        self,
        use_model: bool = True,
        use_fpga: bool = False,
        architecture: Optional[AIEArchitecture] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize benchmark harness.
        
        Args:
            use_model: If True, use AIE performance model for predictions.
            use_fpga: If True, also run actual FPGA execution (requires toolchain + board).
            architecture: Custom FPGA architecture spec for model. Defaults to AIEArchitecture().
            output_dir: Optional directory to save results. Defaults to ./benchmark_results/.
        """
        self.use_model = use_model
        self.use_fpga = use_fpga
        self.architecture = architecture or AIEArchitecture()
        self.model = AIEPerformanceModel(
            architecture=self.architecture, verbose=False
        )
        self.output_dir = Path(output_dir or "benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.presets: Dict[str, ProblemPreset] = {}
        self.results: List[BenchmarkResult] = []

    def add_preset(self, preset: ProblemPreset) -> None:
        """Register a problem preset."""
        self.presets[preset.name] = preset

    def add_presets_from_dict(self, preset_dicts: List[Dict[str, Any]]) -> None:
        """Register presets from list of dicts."""
        for d in preset_dicts:
            preset = ProblemPreset(
                name=d["name"],
                num_variables=d.get("num_variables", 1024),
                num_clauses=d.get("num_clauses", 2048),
                sparsity=d.get("sparsity", 0.5),
                description=d.get("description", ""),
            )
            self.add_preset(preset)

    def run(
        self,
        problem_name: str,
        tile_rows: int = 32,
        tile_cols: int = 32,
        precision_bits: int = 32,
    ) -> BenchmarkResult:
        """Run a single preset problem.
        
        Args:
            problem_name: Name of registered preset.
            tile_rows: AIE tile configuration.
            tile_cols: AIE tile configuration.
            precision_bits: Precision (4, 8, 16, or 32).
        
        Returns:
            BenchmarkResult with predicted/actual times.
        """
        if problem_name not in self.presets:
            raise ValueError(f"Unknown problem: {problem_name}")

        preset = self.presets[problem_name]
        result = BenchmarkResult(
            problem_name=problem_name,
            num_variables=preset.num_variables,
            num_clauses=preset.num_clauses,
            sparsity=preset.sparsity,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            precision_bits=precision_bits,
        )

        try:
            # Model prediction
            if self.use_model:
                cost = self.model.estimate_cost(
                    num_variables=preset.num_variables,
                    num_clauses=preset.num_clauses,
                    sparsity=preset.sparsity,
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                    precision_bits=precision_bits,
                )
                result.predicted_time_ms = cost.total_ms()

            # Compute energy if callable provided
            if preset.energy_fn is not None:
                try:
                    result.energy_value = preset.energy_fn()
                except Exception as e:
                    result.error_msg = f"Energy computation failed: {e}"

            # Actual FPGA execution (if available)
            if self.use_fpga:
                result.actual_time_ms = self._run_fpga_execution(preset)

        except Exception as e:
            result.error_msg = str(e)

        self.results.append(result)
        return result

    def run_all(
        self,
        tile_rows: int = 32,
        tile_cols: int = 32,
        precision_bits: int = 32,
    ) -> List[BenchmarkResult]:
        """Run all registered presets.
        
        Args:
            tile_rows: AIE tile configuration.
            tile_cols: AIE tile configuration.
            precision_bits: Precision (4, 8, 16, or 32).
        
        Returns:
            List of BenchmarkResult for each preset.
        """
        results = []
        for problem_name in self.presets.keys():
            result = self.run(
                problem_name,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                precision_bits=precision_bits,
            )
            results.append(result)

        return results

    def _run_fpga_execution(self, preset: ProblemPreset) -> float:
        """Run actual FPGA execution. (Stub—requires toolchain integration.)
        
        Args:
            preset: Problem to execute.
        
        Returns:
            Actual execution time in milliseconds.
        
        Raises:
            NotImplementedError: FPGA execution not yet integrated.
        """
        # This would integrate with the FPGA harness via:
        # 1. Compile problem via api.define_problem()
        # 2. Load compiled code via BoardRuntime.load()
        # 3. Execute via solve_problem() 
        # 4. Measure elapsed time
        raise NotImplementedError(
            "FPGA execution requires integration with compiled AIE harness. "
            "Use use_model=True for model-only benchmarking."
        )

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save results to JSON.
        
        Args:
            filename: Output filename. Defaults to benchmark_results_<timestamp>.json.
        
        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename

        data = {
            "architecture": self.architecture.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "summary": self._compute_summary(),
            "timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_results(self, filepath: Union[str, Path]) -> None:
        """Load previous benchmark results.
        
        Args:
            filepath: Path to saved JSON results file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.results = [
            BenchmarkResult.from_dict(r) for r in data.get("results", [])
        ]
        arch_dict = data.get("architecture", {})
        self.architecture = AIEArchitecture.from_dict(arch_dict)
        self.model = AIEPerformanceModel(architecture=self.architecture)

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not self.results:
            return {}

        valid_predictions = [r.predicted_time_ms for r in self.results if r.predicted_time_ms is not None]
        valid_actuals = [r.actual_time_ms for r in self.results if r.actual_time_ms is not None]
        errors = [r.accuracy_error_pct() for r in self.results if r.accuracy_error_pct() is not None]

        summary: Dict[str, Any] = {
            "total_runs": len(self.results),
            "successful_runs": sum(1 for r in self.results if r.error_msg is None),
        }

        if valid_predictions:
            summary["avg_predicted_ms"] = sum(valid_predictions) / len(valid_predictions)
            summary["min_predicted_ms"] = min(valid_predictions)
            summary["max_predicted_ms"] = max(valid_predictions)

        if valid_actuals:
            summary["avg_actual_ms"] = sum(valid_actuals) / len(valid_actuals)
            summary["min_actual_ms"] = min(valid_actuals)
            summary["max_actual_ms"] = max(valid_actuals)

        if errors:
            summary["avg_error_pct"] = sum(errors) / len(errors)
            summary["max_error_pct"] = max(errors)

        return summary

    def report(self) -> str:
        """Generate human-readable benchmark report."""
        lines = ["AIE Benchmark Report", "=" * 50]

        if not self.results:
            lines.append("No results.")
            return "\n".join(lines)

        lines.append(f"Architecture: {self.architecture.num_tiles} tiles, "
                     f"{self.architecture.dram_bandwidth_gb_s} GB/s DRAM")
        lines.append(f"Results: {len(self.results)} runs")
        lines.append("")

        for result in self.results:
            lines.append(f"Problem: {result.problem_name}")
            lines.append(f"  Size: {result.num_variables} vars, "
                        f"{result.num_clauses} clauses, {result.sparsity:.1%} sparse")
            if result.predicted_time_ms is not None:
                lines.append(f"  Predicted: {result.predicted_time_ms:.3f} ms")
            if result.actual_time_ms is not None:
                lines.append(f"  Actual: {result.actual_time_ms:.3f} ms")
                lines.append(f"  Error: {result.accuracy_error_pct():.1f}%")
            if result.error_msg:
                lines.append(f"  Error: {result.error_msg}")
            lines.append("")

        # Summary statistics
        summary = self._compute_summary()
        if summary:
            lines.append("Summary Statistics")
            lines.append("-" * 50)
            for key, value in summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# Predefined benchmark suites
def create_small_suite() -> AIEBenchmark:
    """Create benchmark suite with small problems."""
    bench = AIEBenchmark(use_model=True)

    presets = [
        ProblemPreset(
            name="tiny_problem",
            num_variables=64,
            num_clauses=128,
            sparsity=0.3,
            description="Tiny 64-variable QUBO",
        ),
        ProblemPreset(
            name="small_problem",
            num_variables=256,
            num_clauses=512,
            sparsity=0.4,
            description="Small 256-variable QUBO",
        ),
        ProblemPreset(
            name="medium_small",
            num_variables=512,
            num_clauses=1024,
            sparsity=0.5,
            description="Medium-small 512-variable QUBO",
        ),
    ]

    for preset in presets:
        bench.add_preset(preset)

    return bench


def create_standard_suite() -> AIEBenchmark:
    """Create benchmark suite with standard problem sizes."""
    bench = AIEBenchmark(use_model=True)

    presets = [
        ProblemPreset(
            name="standard_1k",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="Standard 1K-variable QUBO",
        ),
        ProblemPreset(
            name="standard_2k",
            num_variables=2048,
            num_clauses=8192,
            sparsity=0.5,
            description="Standard 2K-variable QUBO",
        ),
        ProblemPreset(
            name="standard_4k",
            num_variables=4096,
            num_clauses=16384,
            sparsity=0.5,
            description="Standard 4K-variable QUBO",
        ),
    ]

    for preset in presets:
        bench.add_preset(preset)

    return bench


def create_sparse_suite() -> AIEBenchmark:
    """Create benchmark suite with varying sparsity levels."""
    bench = AIEBenchmark(use_model=True)

    presets = [
        ProblemPreset(
            name="sparse_dense",
            num_variables=1024,
            num_clauses=8192,
            sparsity=0.8,
            description="Dense 1K problem (80% interactions)",
        ),
        ProblemPreset(
            name="sparse_medium",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="Medium 1K problem (50% interactions)",
        ),
        ProblemPreset(
            name="sparse_sparse",
            num_variables=1024,
            num_clauses=1024,
            sparsity=0.125,
            description="Sparse 1K problem (12.5% interactions)",
        ),
        ProblemPreset(
            name="sparse_very_sparse",
            num_variables=1024,
            num_clauses=256,
            sparsity=0.03,
            description="Very sparse 1K problem (3% interactions)",
        ),
    ]

    for preset in presets:
        bench.add_preset(preset)

    return bench


def create_precision_suite() -> AIEBenchmark:
    """Create benchmark suite with varying precision levels."""
    bench = AIEBenchmark(use_model=True)

    presets = [
        ProblemPreset(
            name="precision_fp32",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="1K problem with FP32",
            metadata={"precision_bits": 32},
        ),
        ProblemPreset(
            name="precision_fp16",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="1K problem with FP16",
            metadata={"precision_bits": 16},
        ),
        ProblemPreset(
            name="precision_int8",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="1K problem with INT8",
            metadata={"precision_bits": 8},
        ),
        ProblemPreset(
            name="precision_int4",
            num_variables=1024,
            num_clauses=4096,
            sparsity=0.5,
            description="1K problem with INT4",
            metadata={"precision_bits": 4},
        ),
    ]

    for preset in presets:
        bench.add_preset(preset)

    return bench
