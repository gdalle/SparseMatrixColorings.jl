/*
 * compress_decompress.c — the full round trip: color, compress, decompress.
 *
 *   1. color the columns of the sparsity pattern of A                 (smc_coloring)
 *   2. evaluate one directional derivative per color; stacking them
 *      side by side gives the compressed matrix B = A * S             (smc_compress)
 *   3. scatter B back into the sparse structure of A                  (smc_decompress)
 *
 * Step 2 is done here by the library itself, which lets us check that
 * decompress(compress(A)) == A.  A = tridiag(-1, 2, -1), 6x6, needs 3 colors.
 *
 * Every numerical buffer is passed with its length in elements, and every such
 * length can be queried from the handle before allocating:
 *
 *   nzval  : smc_nnz             -> nnz            elements
 *   Br, Bc : smc_compressed_size -> Br_rows*Br_cols, Bc_rows*Bc_cols elements
 *   A_out  : smc_size            -> m*n            elements
 *
 * A buffer shorter than its minimum is rejected with -3 before a single element
 * is read or written; the last section below demonstrates it.
 *
 * Compile (after building libsmc with juliac — see interfaces/README.md):
 *
 *   gcc -o interfaces/build/compress_decompress \
 *       interfaces/examples/C/compress_decompress.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libsmc.so \
 *       -Wl,-rpath,'$ORIGIN/lib' -lm
 *
 * On macOS use -Wl,-rpath,@loader_path/lib instead.
 *
 * Expected output:
 *   ncolors = 3
 *   pattern: 6 x 6, 16 stored entries
 *   compressed matrix B (6 x 3):
 *      2.00  -1.00   0.00
 *     -1.00   2.00  -1.00
 *     -1.00  -1.00   2.00
 *      2.00  -1.00  -1.00
 *     -1.00   2.00  -1.00
 *      0.00  -1.00   2.00
 *   max error on the 16 stored entries: 0.000e+00
 *   max leakage outside the pattern:    0.000e+00
 *   short nzval -> -3, short Bc -> -3, short A_out -> -3
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "smc.h"

/* A = tridiag(-1, 2, -1), 6x6, compressed sparse column, 0-based.  nzval
 * follows the same ordering as rowval: nzval[k] is the value at row rowval[k]. */

#define M 6
#define N 6
#define NNZ 16

static const int colptr[N + 1] = { 0, 2, 5, 8, 11, 14, 16 };
static const int rowval[NNZ]   = { 0, 1,
                                   0, 1, 2,
                                   1, 2, 3,
                                   2, 3, 4,
                                   3, 4, 5,
                                   4, 5 };
static const double nzval[NNZ] = { 2.0, -1.0,
                                  -1.0,  2.0, -1.0,
                                  -1.0,  2.0, -1.0,
                                  -1.0,  2.0, -1.0,
                                  -1.0,  2.0, -1.0,
                                  -1.0,  2.0 };

int main(void)
{
  /* Defaults: nonsymmetric / column / direct / natural, 0-based, Float64.
     dtype fixes the element type of the compressed and decompressed buffers
     below, and hence the unit of every *_len argument. */
  SmcColoringOptions opts = smc_default_options();

  void *result = NULL;
  int ret = smc_coloring(M, N, colptr, rowval, &opts, &result);
  if (ret != 0) {
    fprintf(stderr, "smc_coloring failed (%d)\n", ret);
    return 1;
  }

  int nc = 0;
  if (smc_ncolors(result, &nc) != 0) {
    fprintf(stderr, "smc_ncolors failed\n");
    smc_result_free(result);
    return 1;
  }
  printf("ncolors = %d\n", nc);

  /* Ask the handle for every size we are about to allocate: the result
     remembers the pattern it was built from, so a caller that only has the
     handle can still size its buffers. */
  int nnz = 0;
  ret = smc_nnz(result, &nnz);
  if (ret != 0) {
    fprintf(stderr, "smc_nnz failed (%d)\n", ret);
    smc_result_free(result);
    return 1;
  }

  int m = 0, n = 0;
  ret = smc_size(result, &m, &n);
  if (ret != 0) {
    fprintf(stderr, "smc_size failed (%d)\n", ret);
    smc_result_free(result);
    return 1;
  }
  printf("pattern: %d x %d, %d stored entries\n", m, n, nnz);

  if (m != M || n != N || nnz != NNZ) {
    fprintf(stderr, "handle describes a different pattern\n");
    smc_result_free(result);
    return 1;
  }

  /* A bidirectional partition splits the compression into a row block Br and a
     column block Bc.  For the column partition used here only Bc is meaningful
     and the Br dimensions come back as 0, so Br is passed as NULL with
     length 0.  A result is bidirectional exactly when Br_cols > 0. */
  int Br_rows = 0, Br_cols = 0, Bc_rows = 0, Bc_cols = 0;
  ret = smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols);
  if (ret != 0) {
    fprintf(stderr, "smc_compressed_size failed (%d)\n", ret);
    smc_result_free(result);
    return 1;
  }

  /* Element counts are computed in size_t: m*n overflows a 32-bit int at
     perfectly ordinary dimensions. */
  size_t Bc_len = (size_t)Bc_rows * (size_t)Bc_cols;
  size_t A_len  = (size_t)m * (size_t)n;

  double *Bc = (double *)calloc(Bc_len, sizeof(double));
  double *Ad = (double *)calloc(A_len, sizeof(double));
  if (Bc == NULL || Ad == NULL) {
    fprintf(stderr, "out of memory\n");
    free(Bc); free(Ad);
    smc_result_free(result);
    return 1;
  }

  /* Compress: B[:, c] = sum of the columns of A colored c.  In a real AD code
     this buffer would instead be filled by evaluating one directional
     derivative per color. */
  ret = smc_compress(result,
                     nzval, (size_t)nnz,
                     NULL, 0,           /* Br unused by a column partition */
                     Bc, Bc_len);
  if (ret != 0) {
    fprintf(stderr, "smc_compress failed (%d)\n", ret);
    free(Bc); free(Ad);
    smc_result_free(result);
    return 1;
  }

  /* Dense buffers are column-major: B[i,j] lives at Bc[i + j*Bc_rows]. */
  printf("compressed matrix B (%d x %d):\n", Bc_rows, Bc_cols);
  for (int i = 0; i < Bc_rows; i++) {
    for (int j = 0; j < Bc_cols; j++)
      printf("  %5.2f", Bc[i + j * Bc_rows]);
    printf("\n");
  }

  /* Decompress: recover the full m-by-n dense A from B, again column-major. */
  ret = smc_decompress(result,
                       NULL, 0,         /* Br unused by a column partition */
                       Bc, Bc_len,
                       Ad, A_len);
  if (ret != 0) {
    fprintf(stderr, "smc_decompress failed (%d)\n", ret);
    free(Bc); free(Ad);
    smc_result_free(result);
    return 1;
  }

  /* Check that every stored entry came back exactly and that nothing leaked
     into positions outside the sparsity pattern. */
  double max_err = 0.0;
  char *in_pattern = (char *)calloc(A_len, sizeof(char));
  if (in_pattern == NULL) {
    fprintf(stderr, "out of memory\n");
    free(Bc); free(Ad);
    smc_result_free(result);
    return 1;
  }

  for (int j = 0; j < N; j++) {
    for (int k = colptr[j]; k < colptr[j + 1]; k++) {
      int i = rowval[k];
      in_pattern[i + j * M] = 1;
      double err = fabs(Ad[i + j * M] - nzval[k]);
      if (err > max_err) max_err = err;
    }
  }
  printf("max error on the %d stored entries: %.3e\n", NNZ, max_err);

  double max_leak = 0.0;
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      if (!in_pattern[i + j * M]) {
        double leak = fabs(Ad[i + j * M]);
        if (leak > max_leak) max_leak = leak;
      }
    }
  }
  printf("max leakage outside the pattern:    %.3e\n", max_leak);

  /* Understating any length is refused with -3, and nothing is read or written
     before the refusal — the guarantee that lets a handle be reused safely. */
  int short_nzval = smc_compress(result, nzval, (size_t)nnz - 1, NULL, 0, Bc, Bc_len);
  int short_Bc    = smc_compress(result, nzval, (size_t)nnz, NULL, 0, Bc, Bc_len - 1);
  int short_A     = smc_decompress(result, NULL, 0, Bc, Bc_len, Ad, A_len - 1);
  printf("short nzval -> %d, short Bc -> %d, short A_out -> %d\n",
         short_nzval, short_Bc, short_A);

  free(in_pattern);
  free(Bc);
  free(Ad);
  smc_result_free(result);

  /* Direct decompression is an exact scatter, so both errors must be zero. */
  if (max_err != 0.0 || max_leak != 0.0) {
    fprintf(stderr, "round trip did not reproduce A\n");
    return 1;
  }

  if (short_nzval != -3 || short_Bc != -3 || short_A != -3) {
    fprintf(stderr, "a short buffer was not rejected\n");
    return 1;
  }

  return 0;
}
