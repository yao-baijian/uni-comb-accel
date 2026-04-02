// SBM-oriented AIE kernel helpers.
//
// This translation unit keeps the sparse matrix-vector multiply and the time
// evolution update in a single kernel-oriented C++ file so ARIES-oriented
// flows can reuse them directly during simulation or code generation.

#include <algorithm>
#include <cmath>

static constexpr int kHeaderStride = 7;

extern "C" void sbm_spmv_tcsr(
    const int *tile_headers,
    const float *values,
    const int *col_indices,
    const int *row_ptr,
    const float *vec_in,
    float *vec_out,
    int num_rows) {
  for (int r = 0; r < num_rows; ++r) {
    vec_out[r] = 0.0f;
  }

  constexpr int kChunk = 16;
  float xbuf0[kChunk];
  float xbuf1[kChunk];

  int tile_idx = 0;
  while (true) {
    const int base = tile_idx * kHeaderStride;
    const int row_start = tile_headers[base + 2];
    const int row_end = tile_headers[base + 3];
    const int value_offset = tile_headers[base + 4];
    const int row_ptr_offset = tile_headers[base + 5];
    const int nnz = tile_headers[base + 6];

    if (tile_idx > 0 && row_start == 0 && row_end == 0 && nnz == 0) {
      break;
    }
    if (nnz == 0 && row_start >= num_rows) {
      break;
    }

    for (int r = row_start; r < row_end; ++r) {
      const int local_row = r - row_start;
      const int begin = row_ptr[row_ptr_offset + local_row];
      const int end = row_ptr[row_ptr_offset + local_row + 1];
      float acc = vec_out[r];

      int p = begin;
      bool use_buf0 = true;
      while (p < end) {
        const int chunk_end = std::min(p + kChunk, end);

        float *buf = use_buf0 ? xbuf0 : xbuf1;
        for (int q = p; q < chunk_end; ++q) {
          buf[q - p] = vec_in[col_indices[value_offset + q]];
        }

        for (int q = p; q < chunk_end; ++q) {
          acc += values[value_offset + q] * buf[q - p];
        }

        use_buf0 = !use_buf0;
        p = chunk_end;
      }

      vec_out[r] = acc;
    }

    ++tile_idx;
  }
}

extern "C" void sbm_te(
    float *x_comp,
    float *y_comp,
    const float *spmv_out,
    int num_vars,
    float alpha,
    float xi,
    float dt) {
  for (int i = 0; i < num_vars; ++i) {
    y_comp[i] += ((-1.0f + alpha) * x_comp[i] + xi * spmv_out[i]) * dt;
    x_comp[i] += y_comp[i] * dt;

    if (std::fabs(x_comp[i]) > 1.0f) {
      y_comp[i] = 0.0f;
      if (x_comp[i] > 1.0f) {
        x_comp[i] = 1.0f;
      } else if (x_comp[i] < -1.0f) {
        x_comp[i] = -1.0f;
      }
    }
  }
}