"""Performance modeling helpers for CSR/TCSR-style SpMV on AIE-like tiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SpMVArchitecture:
    """Architecture parameters that drive one-iteration SpMV throughput."""

    name: str
    freq_mhz: float
    tile_stream_bandwidth_bits_per_cycle: int
    value_bit_width: int
    index_bit_width: int
    vector_bit_width: int
    vector_reads_per_cycle: int
    vector_writes_per_cycle: int = 0
    macs_per_cycle: int = 1


@dataclass(frozen=True)
class GsetMatrixStats:
    """Minimal matrix stats extracted from a Gset edge-list file."""

    num_rows: int
    num_edges: int
    nnz_for_spmv: int


@dataclass(frozen=True)
class SpMVIterationEstimate:
    """Estimated cycle/time breakdown for one SpMV iteration."""

    architecture: str
    matrix_path: str
    num_rows: int
    nnz: int
    matrix_stream_cycles: float
    vector_read_cycles: float
    vector_write_cycles: float
    compute_cycles: float
    total_cycles: float
    estimated_time_us: float


def parse_gset_stats(path: str | Path, *, symmetric: bool = True) -> GsetMatrixStats:
    """Parse `benchmarks/Gset/G*` file and derive nnz used by SpMV."""

    p = Path(path)
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty Gset file: {p}")

    head = lines[0].split()
    if len(head) < 2:
        raise ValueError(f"Invalid Gset header in {p}: {lines[0]!r}")

    n = int(head[0])
    m = int(head[1])
    nnz = int(2 * m if symmetric else m)
    return GsetMatrixStats(num_rows=n, num_edges=m, nnz_for_spmv=nnz)


def estimate_spmv_iteration(
    matrix_path: str | Path,
    arch: SpMVArchitecture,
    *,
    symmetric: bool = True,
) -> SpMVIterationEstimate:
    """Estimate one SpMV iteration latency from architecture + matrix stats.

    Model assumptions:
    - matrix values/indices are streamed from DDR to tile stream fabric;
    - current vector is in tile local RAM (ping-pong with next vector);
    - one multiply-accumulate per non-zero.
    """

    stats = parse_gset_stats(matrix_path, symmetric=symmetric)
    nnz = float(stats.nnz_for_spmv)
    rows = float(stats.num_rows)

    bits_per_nnz_stream = float(arch.value_bit_width + arch.index_bit_width)
    rowptr_bits = float((stats.num_rows + 1) * arch.index_bit_width)
    matrix_bits_total = nnz * bits_per_nnz_stream + rowptr_bits

    matrix_stream_cycles = matrix_bits_total / float(arch.tile_stream_bandwidth_bits_per_cycle)
    vector_read_cycles = nnz / float(max(arch.vector_reads_per_cycle, 1))
    vector_write_cycles = 0.0
    if arch.vector_writes_per_cycle > 0:
        vector_write_cycles = rows / float(arch.vector_writes_per_cycle)

    compute_cycles = nnz / float(max(arch.macs_per_cycle, 1))
    total_cycles = max(matrix_stream_cycles, vector_read_cycles, vector_write_cycles, compute_cycles)
    time_us = total_cycles / float(arch.freq_mhz)

    return SpMVIterationEstimate(
        architecture=arch.name,
        matrix_path=str(matrix_path),
        num_rows=stats.num_rows,
        nnz=stats.nnz_for_spmv,
        matrix_stream_cycles=matrix_stream_cycles,
        vector_read_cycles=vector_read_cycles,
        vector_write_cycles=vector_write_cycles,
        compute_cycles=compute_cycles,
        total_cycles=total_cycles,
        estimated_time_us=time_us,
    )


def estimate_multiple_architectures(
    matrix_path: str | Path,
    architectures: Dict[str, SpMVArchitecture],
    *,
    symmetric: bool = True,
) -> Dict[str, SpMVIterationEstimate]:
    """Run one-iteration SpMV estimation for each architecture."""

    out: Dict[str, SpMVIterationEstimate] = {}
    for name, arch in architectures.items():
        out[name] = estimate_spmv_iteration(matrix_path, arch, symmetric=symmetric)
    return out
