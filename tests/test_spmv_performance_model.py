"""Performance-model regression test for one-iteration SpMV on Gset matrices."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backend.spmv_performance_model import (
    SpMVArchitecture,
    estimate_multiple_architectures,
)


def test_gset_one_iteration_time_across_architectures() -> None:
    """Estimate one-iteration latency for same Gset matrix under different architectures."""

    matrix = REPO_ROOT / "benchmarks" / "Gset" / "G1"
    assert matrix.exists(), f"Missing benchmark matrix: {matrix}"

    architectures = {
        "narrow": SpMVArchitecture(
            name="narrow",
            freq_mhz=1000.0,
            tile_stream_bandwidth_bits_per_cycle=256,
            value_bit_width=16,
            index_bit_width=16,
            vector_bit_width=16,
            vector_reads_per_cycle=4,
            vector_writes_per_cycle=4,
            macs_per_cycle=4,
        ),
        "balanced": SpMVArchitecture(
            name="balanced",
            freq_mhz=1000.0,
            tile_stream_bandwidth_bits_per_cycle=512,
            value_bit_width=16,
            index_bit_width=16,
            vector_bit_width=16,
            vector_reads_per_cycle=8,
            vector_writes_per_cycle=8,
            macs_per_cycle=8,
        ),
        "wide": SpMVArchitecture(
            name="wide",
            freq_mhz=1000.0,
            tile_stream_bandwidth_bits_per_cycle=1024,
            value_bit_width=16,
            index_bit_width=16,
            vector_bit_width=16,
            vector_reads_per_cycle=16,
            vector_writes_per_cycle=16,
            macs_per_cycle=16,
        ),
    }

    estimates = estimate_multiple_architectures(matrix, architectures, symmetric=True)

    narrow_us = estimates["narrow"].estimated_time_us
    balanced_us = estimates["balanced"].estimated_time_us
    wide_us = estimates["wide"].estimated_time_us

    # Print for quick manual inspection in CI logs/local runs.
    print(f"G1 one-iter estimate narrow   : {narrow_us:.3f} us")
    print(f"G1 one-iter estimate balanced : {balanced_us:.3f} us")
    print(f"G1 one-iter estimate wide     : {wide_us:.3f} us")

    assert narrow_us > balanced_us > wide_us
