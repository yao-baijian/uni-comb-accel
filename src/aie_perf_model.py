"""AIE execution-time performance model for QUBO/Ising problems.

Provides cost modeling for AIE-compiled energy problems based on:
- FPGA architecture parameters (tiles, DRAM, compute units, bandwidth)
- Problem characteristics (operation counts, data movement)
- AIE configuration (tile sizes, precision)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass
class AIEArchitecture:
    """FPGA architecture parameters for performance modeling.
    
    Attributes:
        num_tiles: Total number of AIE tiles available.
        dram_size_mb: Total DRAM capacity in MB.
        dram_bandwidth_gb_s: Peak DRAM bandwidth in GB/s.
        pe_per_tile: Processing elements per tile.
        compute_units_per_pe: Compute units (ALUs, multipliers) per PE.
        l1_cache_kb: L1 cache per tile in KB.
        l2_bandwidth_gb_s: Inter-tile L2 bandwidth in GB/s.
        tile_frequency_ghz: Tile operating frequency in GHz.
    """

    num_tiles: int = 16
    dram_size_mb: int = 128
    dram_bandwidth_gb_s: float = 16.0
    pe_per_tile: int = 4
    compute_units_per_pe: int = 4
    l1_cache_kb: int = 32
    l2_bandwidth_gb_s: float = 32.0
    tile_frequency_ghz: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> AIEArchitecture:
        """Deserialize from dictionary."""
        return AIEArchitecture(**d)


@dataclass
class OperationCost:
    """Cost breakdown for an operation execution phase."""

    initialization_ms: float = 0.0
    compute_ms: float = 0.0
    memory_access_ms: float = 0.0
    communication_ms: float = 0.0
    finalization_ms: float = 0.0

    def total_ms(self) -> float:
        """Total execution time in milliseconds."""
        return (
            self.initialization_ms
            + self.compute_ms
            + self.memory_access_ms
            + self.communication_ms
            + self.finalization_ms
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return asdict(self)


class AIEPerformanceModel:
    """Cost model for AIE-compiled energy problems.
    
    Estimates execution time based on:
    1. Operation counts extracted from problem characteristics
    2. Data movement patterns inferred from problem structure
    3. Tile utilization based on AIE configuration
    4. Memory bandwidth and latency factors
    
    Example:
        >>> arch = AIEArchitecture(num_tiles=16, dram_bandwidth_gb_s=16.0)
        >>> model = AIEPerformanceModel(arch)
        
        >>> # Estimate cost for a 1024-variable QUBO problem
        >>> cost = model.estimate_cost(
        ...     num_variables=1024,
        ...     num_clauses=4096,
        ...     sparsity=0.2,
        ...     tile_rows=32,
        ...     tile_cols=32,
        ...     precision_bits=32
        ... )
        >>> print(f"Estimated time: {cost.total_ms():.2f} ms")
    """

    def __init__(
        self,
        architecture: Optional[AIEArchitecture] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize performance model.
        
        Args:
            architecture: FPGA architecture spec. Defaults to typical AIE values.
            verbose: If True, print intermediate cost calculations.
        """
        self.arch = architecture or AIEArchitecture()
        self.verbose = verbose

    def estimate_cost(
        self,
        num_variables: int,
        num_clauses: int,
        sparsity: float = 0.5,
        tile_rows: int = 32,
        tile_cols: int = 32,
        precision_bits: int = 32,
        problem_metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationCost:
        """Estimate execution time for a QUBO/Ising problem.
        
        Args:
            num_variables: Number of Ising variables (nodes).
            num_clauses: Number of clauses/interactions (edges).
            sparsity: Sparsity factor (0.0-1.0) of interaction matrix.
            tile_rows: Number of rows in AIE sparse tiling.
            tile_cols: Number of cols in AIE sparse tiling.
            precision_bits: Precision per variable (32, 16, 8, or 4).
            problem_metadata: Optional dict with additional problem info.
        
        Returns:
            OperationCost with execution-time breakdown.
        """
        # Clamp inputs
        sparsity = max(0.0, min(1.0, sparsity))
        precision_bits = max(4, min(32, precision_bits))
        num_clauses = max(1, min(num_variables * num_variables, num_clauses))

        # Phase 1: Initialization & data loading
        init_cost = self._estimate_initialization_cost(
            num_variables, num_clauses, precision_bits
        )

        # Phase 2: Main computation cost
        compute_cost = self._estimate_compute_cost(
            num_variables,
            num_clauses,
            sparsity,
            tile_rows,
            tile_cols,
            precision_bits,
        )

        # Phase 3: Memory access overhead
        memory_cost = self._estimate_memory_cost(
            num_variables, num_clauses, sparsity, precision_bits
        )

        # Phase 4: Inter-tile communication
        comm_cost = self._estimate_communication_cost(
            num_variables, num_clauses, tile_rows, tile_cols
        )

        # Phase 5: Finalization & output
        final_cost = self._estimate_finalization_cost(num_variables, precision_bits)

        return OperationCost(
            initialization_ms=init_cost,
            compute_ms=compute_cost,
            memory_access_ms=memory_cost,
            communication_ms=comm_cost,
            finalization_ms=final_cost,
        )

    def _estimate_initialization_cost(
        self,
        num_variables: int,
        num_clauses: int,
        precision_bits: int,
    ) -> float:
        """Estimate time to load problem & initialize state."""
        # DMA transfer for problem weights + biases
        data_size_bytes = (
            num_variables * (precision_bits // 8)  # Variable state
            + num_clauses * (2 * (precision_bits // 8))  # Clause weights & indices
        )
        dram_latency_ms = 0.5  # L1 miss + DRAM access
        transfer_ms = data_size_bytes / (self.arch.dram_bandwidth_gb_s * 1e9)

        # Tile setup (synchronization, barrier initialization)
        setup_ms = 0.1 * self.arch.num_tiles

        return dram_latency_ms + transfer_ms + setup_ms

    def _estimate_compute_cost(
        self,
        num_variables: int,
        num_clauses: int,
        sparsity: float,
        tile_rows: int,
        tile_cols: int,
        precision_bits: int,
    ) -> float:
        """Estimate core QUBO/Ising computation time."""
        # Effective interactions after sparsity
        effective_clauses = int(num_clauses * sparsity)

        # Operations per clause: 1 multiply + 1 accumulate (FMA)
        # Plus reduction over all iterations
        ops_per_clause = 2  # Simplified: actual ALU ops

        # Tile utilization factor (not all tiles may be fully loaded)
        active_tiles = min(
            self.arch.num_tiles,
            math.ceil(effective_clauses / max(tile_rows * tile_cols, 1)),
        )
        utilization = active_tiles / max(self.arch.num_tiles, 1)

        # FLOPs available per cycle
        peak_flops_per_cycle = (
            active_tiles
            * self.arch.pe_per_tile
            * self.arch.compute_units_per_pe
        )

        # Total FLOPs needed
        total_flops = effective_clauses * ops_per_clause * (
            1 + math.log2(max(1, num_variables // 8))
        )  # Tree reduction

        # Cycles needed (with reduced precision overhead)
        precision_factor = 1.0 if precision_bits == 32 else (precision_bits / 32.0) * 0.8
        cycles = total_flops / max(peak_flops_per_cycle * precision_factor, 1)

        # Convert cycles to ms
        compute_ms = cycles / (self.arch.tile_frequency_ghz * 1e3)

        # Overhead factor for scheduling & pipeline stalls
        overhead = 1.15

        if self.verbose:
            print(
                f"Compute: {effective_clauses} clauses, {active_tiles}/{self.arch.num_tiles} tiles, "
                f"{cycles:.0f} cycles → {compute_ms * overhead:.3f} ms"
            )

        return compute_ms * overhead

    def _estimate_memory_cost(
        self,
        num_variables: int,
        num_clauses: int,
        sparsity: float,
        precision_bits: int,
    ) -> float:
        """Estimate memory access stalls & cache misses."""
        # L1 cache miss rate estimation
        working_set_bytes = num_variables * (precision_bits // 8)
        cache_capacity_bytes = self.arch.l1_cache_kb * 1024

        if working_set_bytes <= cache_capacity_bytes:
            miss_rate = 0.05  # Very low miss rate
        elif working_set_bytes <= self.arch.dram_size_mb * 1e6:
            # Estimate miss rate from conflict/capacity misses
            miss_rate = min(0.3, working_set_bytes / (cache_capacity_bytes * 1.5))
        else:
            miss_rate = 0.8  # Working set doesn't fit

        # Miss penalty in cycles
        miss_penalty_cycles = 10  # Typical L1→L2 + DRAM latency

        # Effective clauses & memory ops
        effective_clauses = int(num_clauses * sparsity)
        memory_ops = effective_clauses * 2  # Read weight + read/write accumulator

        # Stall cycles due to misses
        stall_cycles = memory_ops * miss_rate * miss_penalty_cycles

        memory_ms = stall_cycles / (self.arch.tile_frequency_ghz * 1e3)

        if self.verbose:
            print(
                f"Memory: miss_rate={miss_rate:.2%}, stall_cycles={stall_cycles:.0f} → {memory_ms:.3f} ms"
            )

        return memory_ms

    def _estimate_communication_cost(
        self,
        num_variables: int,
        num_clauses: int,
        tile_rows: int,
        tile_cols: int,
    ) -> float:
        """Estimate inter-tile communication overhead."""
        # Estimate number of tile boundaries crossed
        tiles_per_dim = math.ceil(math.sqrt(self.arch.num_tiles))
        avg_hops = tiles_per_dim  # Average routing hops

        # Data flowing between tiles
        boundary_data_bytes = (
            int(num_clauses * (16 // 8)) * avg_hops
        )  # Simplified: header per clause

        # Stream latency for inter-tile traffic
        stream_ms = boundary_data_bytes / (self.arch.l2_bandwidth_gb_s * 1e9)

        # Synchronization overhead between phases
        sync_barriers = 3  # init, compute, finalize
        sync_ms = sync_barriers * 0.05

        comm_ms = stream_ms + sync_ms

        if self.verbose:
            print(
                f"Communication: {boundary_data_bytes} bytes, {avg_hops} hops → {comm_ms:.3f} ms"
            )

        return comm_ms

    def _estimate_finalization_cost(
        self,
        num_variables: int,
        precision_bits: int,
    ) -> float:
        """Estimate result aggregation & write-back time."""
        # Final result size (state + energy)
        result_size_bytes = num_variables * (precision_bits // 8) + 64  # 64 bytes for metadata

        # Write back to DRAM
        writeout_ms = result_size_bytes / (self.arch.dram_bandwidth_gb_s * 1e9)

        # Barrier & cleanup
        cleanup_ms = 0.05

        return writeout_ms + cleanup_ms

    def estimate_batch_cost(
        self,
        problems: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Estimate cost for a batch of problems.
        
        Args:
            problems: List of problem dicts, each with keys:
                'num_variables', 'num_clauses', 'sparsity', 'tile_rows', 'tile_cols', 'precision_bits'.
        
        Returns:
            Dict with per-problem costs and aggregates.
        """
        costs = []
        for problem in problems:
            cost = self.estimate_cost(
                num_variables=problem.get("num_variables", 1024),
                num_clauses=problem.get("num_clauses", 4096),
                sparsity=problem.get("sparsity", 0.5),
                tile_rows=problem.get("tile_rows", 32),
                tile_cols=problem.get("tile_cols", 32),
                precision_bits=problem.get("precision_bits", 32),
            )
            costs.append(cost)

        total_ms = sum(c.total_ms() for c in costs)
        avg_ms = total_ms / len(costs) if costs else 0.0

        return {
            "per_problem": [c.to_dict() for c in costs],
            "total_ms": total_ms,
            "average_ms": avg_ms,
            "num_problems": len(costs),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "architecture": self.arch.to_dict(),
            "class": "AIEPerformanceModel",
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> AIEPerformanceModel:
        """Deserialize model from dictionary."""
        arch = AIEArchitecture.from_dict(d.get("architecture", {}))
        return AIEPerformanceModel(architecture=arch)


# Convenience function for quick cost estimation
def estimate_aie_time(
    num_variables: int,
    num_clauses: int,
    sparsity: float = 0.5,
    tile_rows: int = 32,
    tile_cols: int = 32,
    precision_bits: int = 32,
    architecture: Optional[AIEArchitecture] = None,
) -> float:
    """Quick function to estimate AIE execution time in milliseconds.
    
    Args:
        num_variables: Number of Ising variables.
        num_clauses: Number of interactions/clauses.
        sparsity: Sparsity factor (0-1).
        tile_rows: AIE tile rows.
        tile_cols: AIE tile cols.
        precision_bits: Precision (4, 8, 16, or 32).
        architecture: Optional custom FPGA architecture.
    
    Returns:
        Estimated execution time in milliseconds.
    """
    model = AIEPerformanceModel(architecture=architecture)
    cost = model.estimate_cost(
        num_variables=num_variables,
        num_clauses=num_clauses,
        sparsity=sparsity,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        precision_bits=precision_bits,
    )
    return cost.total_ms()
