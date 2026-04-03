// AIE-oriented SpMV kernel over standard CSR layout.
// This version is C++-valid and suitable for host/x86 functional simulation.

extern "C" void spmv_csr(
    const float *values,
    const int *col_indices,
    const int *row_ptr,
    const float *vec_in,
    float *vec_out,
    int num_rows) {
  for (int r = 0; r < num_rows; ++r) {
    const int begin = row_ptr[r];
    const int end = row_ptr[r + 1];

    float acc = 0.0f;
    for (int p = begin; p < end; ++p) {
      acc += values[p] * vec_in[col_indices[p]];
    }
    vec_out[r] = acc;
  }
}
