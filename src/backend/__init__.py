"""Backend modules for code generation and sparse lowering."""

from .precision import Precision, normalize_precision
from .sparse_formats import SparseFormat, normalize_sparse_format
from .csr import CSRData, generate_csr
from .spmv_performance_model import (
	GsetMatrixStats,
	SpMVArchitecture,
	SpMVIterationEstimate,
	estimate_multiple_architectures,
	estimate_spmv_iteration,
	parse_gset_stats,
)
