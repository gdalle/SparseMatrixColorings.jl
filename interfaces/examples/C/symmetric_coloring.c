/*
 * symmetric_coloring.c — color a symmetric matrix (Hessian-style).
 *
 * A is the 2D 5-point Laplacian on a 4x4 grid: a 16x16 symmetric matrix.  The
 * difference with basic_coloring.c is opts.structure = SMC_SYMMETRIC, which
 * exploits A[i,j] being readable from either column j or column i and so needs
 * fewer colors.  With SMC_DIRECT this is a star coloring: every entry is
 * recovered by a plain copy, where SMC_SUBSTITUTION would solve instead.
 *
 * Compile (after building libsmc with juliac — see interfaces/README.md):
 *
 *   gcc -o interfaces/build/symmetric_coloring \
 *       interfaces/examples/C/symmetric_coloring.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libsmc.so \
 *       -Wl,-rpath,'$ORIGIN/lib'
 *
 * On macOS use -Wl,-rpath,@loader_path/lib instead.
 *
 * Expected output:
 *   grid 4x4  ->  n = 16, nnz = 64
 *   ncolors = 5
 *   color of each grid node:
 *     1 2 1 3
 *     3 1 4 1
 *     1 5 1 2
 *     2 1 3 1
 */

#include <stdio.h>
#include <stdlib.h>

#include "smc.h"

#define NX 4                 /* grid points in x             */
#define NY 4                 /* grid points in y             */
#define N (NX * NY)          /* matrix order                 */
#define MAX_NNZ (5 * N)      /* at most 5 entries per column */

/* Node (i,j) of the grid, 0 <= i < NX, 0 <= j < NY, maps to column i + j*NX. */
static int node(int i, int j) { return i + j * NX; }

/* Build the 5-point Laplacian in CSC form (0-based), returning the number of
 * stored entries.  Column k holds k-NX, k-1, k, k+1, k+NX, already sorted in
 * increasing order as CSC requires. */
static int build_laplacian(int *colptr, int *rowval)
{
  int nnz = 0;

  for (int j = 0; j < NY; j++) {
    for (int i = 0; i < NX; i++) {
      int k = node(i, j);
      colptr[k] = nnz;

      if (j > 0)      rowval[nnz++] = node(i, j - 1);  /* south */
      if (i > 0)      rowval[nnz++] = node(i - 1, j);  /* west  */
      rowval[nnz++] = k;                               /* centre */
      if (i < NX - 1) rowval[nnz++] = node(i + 1, j);  /* east  */
      if (j < NY - 1) rowval[nnz++] = node(i, j + 1);  /* north */
    }
  }
  colptr[N] = nnz;

  return nnz;
}

int main(void)
{
  int colptr[N + 1];
  int rowval[MAX_NNZ];

  int nnz = build_laplacian(colptr, rowval);
  printf("grid %dx%d  ->  n = %d, nnz = %d\n", NX, NY, N, nnz);

  /* symmetric_pattern = 1 asserts that the pattern really is symmetric, letting
     the library skip building the transposed pattern.  It is trusted, not
     checked, so only set it when it holds — as it does for a Laplacian.
     The partition stays SMC_COLUMN: symmetric problems are colored by column. */
  SmcColoringOptions opts = smc_default_options();
  opts.structure         = SMC_SYMMETRIC;
  opts.partition         = SMC_COLUMN;
  opts.decompression     = SMC_DIRECT;
  opts.symmetric_pattern = 1;

  void *result = NULL;
  int ret = smc_coloring(N, N, colptr, rowval, &opts, &result);
  if (ret != 0) {
    fprintf(stderr, "smc_coloring failed (%d)\n", ret);
    return 1;
  }

  int nc = 0;
  ret = smc_ncolors(result, &nc);
  if (ret != 0) {
    fprintf(stderr, "smc_ncolors failed (%d)\n", ret);
    smc_result_free(result);
    return 1;
  }
  printf("ncolors = %d\n", nc);

  /* Printing the colors laid out on the grid makes the pattern visible. */
  int colors[N];
  ret = smc_column_colors(result, colors, N);
  if (ret != 0) {
    fprintf(stderr, "smc_column_colors failed (%d)\n", ret);
    smc_result_free(result);
    return 1;
  }

  printf("color of each grid node:\n");
  for (int j = 0; j < NY; j++) {
    printf("  ");
    for (int i = 0; i < NX; i++)
      printf(" %d", colors[node(i, j)]);
    printf("\n");
  }

  smc_result_free(result);

  /* Sanity check: every color is a valid label in 1..ncolors. */
  for (int k = 0; k < N; k++) {
    if (colors[k] < 1 || colors[k] > nc) {
      fprintf(stderr, "column %d has out-of-range color %d\n", k, colors[k]);
      return 1;
    }
  }

  return 0;
}
