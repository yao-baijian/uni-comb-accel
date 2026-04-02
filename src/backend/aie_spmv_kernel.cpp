// AIE-oriented SpMV kernel over TCSR data layout.
// This version is valid C++ and can be used for x86 functional simulation.

#include <algorithm>
#include <cstddef>

// Tile header layout (7 ints each):
// [tile_row, tile_col, row_start, row_end, value_offset, row_ptr_offset, nnz]
static constexpr int kHeaderStride = 7;

extern "C" void spmv_tcsr(
    const int *tile_headers,
    const float *values,
    const int *col_indices,
    const int *row_ptr,
    const float *vec_in,
    float *vec_out,
    int num_rows) {
  // Initialize output once. Each tile accumulates partial sums.
  for (int r = 0; r < num_rows; ++r) {
    vec_out[r] = 0.0f;
  }

  // Double-buffering metadata for vec_in fetch simulation.
  // On real AIE codegen this maps to local memory ping-pong buffers.
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

    // Sentinel: all zeros header marks end of tile list.
    if (tile_idx > 0 && row_start == 0 && row_end == 0 && nnz == 0) {
      break;
    }
    if (nnz == 0 && row_start >= num_rows) {
      break;
    }

    // Per-row traversal inside this tile.
    // Compiler can vectorize inner multiply-accumulate (VLIW/vector unit friendly).
    for (int r = row_start; r < row_end; ++r) {
      const int local_row = r - row_start;
      const int begin = row_ptr[row_ptr_offset + local_row];
      const int end = row_ptr[row_ptr_offset + local_row + 1];
      float acc = vec_out[r];

      int p = begin;
      bool use_buf0 = true;
      while (p < end) {
        const int chunk_end = std::min(p + kChunk, end);

        // Stage loads into a local buffer to hide latency on AIE memory hierarchy.
        float *buf = use_buf0 ? xbuf0 : xbuf1;
        for (int q = p; q < chunk_end; ++q) {
          buf[q - p] = vec_in[col_indices[value_offset + q]];
        }

        // Compute on the previously prepared chunk.
        // This loop is intentionally flat for auto-vectorization.
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
