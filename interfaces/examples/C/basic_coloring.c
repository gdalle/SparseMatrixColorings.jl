/*
 * basic_coloring.c — minimal example: color the columns of a sparse matrix.
 *
 * A is the 6x6 tridiagonal matrix tridiag(-1, 2, -1).  Two columns may share a
 * color only if they have no nonzero in a common row.  Only the sparsity
 * pattern is used; see compress_decompress.c for the numerical values.
 *
 * Compile (after building libsmc with juliac — see interfaces/README.md):
 *
 *   gcc -o interfaces/build/basic_coloring interfaces/examples/C/basic_coloring.c \
 *       -I interfaces/build/include \
 *       interfaces/build/lib/libsmc.so \
 *       -Wl,-rpath,'$ORIGIN/lib'
 *
 * The rpath assumes the binary sits in interfaces/build/, next to lib/; on
 * macOS use -Wl,-rpath,@loader_path/lib instead.
 *
 * Expected output:
 *   SparseMatrixColorings 0.4.x
 *   ncolors = 3
 *   column colors = [ 1 2 3 1 2 3 ]
 *   group 1 = { 0 3 }
 *   group 2 = { 1 4 }
 *   group 3 = { 2 5 }
 */

#include <stdio.h>
#include <stdlib.h>

#include "smc.h"

/* A = tridiag(-1, 2, -1), 6x6, compressed sparse column.  With the default
 * opts.index_base == 0 both colptr and rowval are 0-based: column j occupies
 * entries colptr[j] .. colptr[j+1] - 1, and rowval[k] is the row of entry k. */

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

int main(void)
{
  int major, minor, patch;
  smc_version(&major, &minor, &patch);
  printf("SparseMatrixColorings %d.%d.%d\n", major, minor, patch);

  /* Defaults: nonsymmetric / column / direct / natural, 0-based, Float64. */
  SmcColoringOptions opts = smc_default_options();

  /* The result is an opaque handle owned by the library; it must be released
     with smc_result_free(). */
  void *result = NULL;
  int ret = smc_coloring(M, N, colptr, rowval, &opts, &result);
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

  /* Colors are labels in 1..ncolors, not indices: they are never shifted by
     opts.index_base.  The value 0 is the neutral color and appears only when
     opts.postprocessing is enabled.  The buffer must hold at least n entries. */
  int *colors = (int *)malloc(sizeof(int) * N);
  if (colors == NULL) {
    fprintf(stderr, "out of memory\n");
    smc_result_free(result);
    return 1;
  }

  ret = smc_column_colors(result, colors, N);
  if (ret != 0) {
    fprintf(stderr, "smc_column_colors failed (%d)\n", ret);
    free(colors);
    smc_result_free(result);
    return 1;
  }

  printf("column colors = [");
  for (int j = 0; j < N; j++)
    printf(" %d", colors[j]);
  printf(" ]\n");

  /* The same coloring seen as groups of columns sharing a color.  Group indices
     run over 1..ncolumn_groups; the members are column indices in the caller's
     index base.  Sizes are queried first to size the buffer exactly. */
  int ngroups = 0;
  ret = smc_ncolumn_groups(result, &ngroups);
  if (ret != 0) {
    fprintf(stderr, "smc_ncolumn_groups failed (%d)\n", ret);
    free(colors);
    smc_result_free(result);
    return 1;
  }

  for (int g = 1; g <= ngroups; g++) {
    int size = 0;
    if (smc_column_group_size(result, g, &size) != 0) {
      fprintf(stderr, "smc_column_group_size failed for group %d\n", g);
      free(colors);
      smc_result_free(result);
      return 1;
    }

    int *members = (int *)malloc(sizeof(int) * (size > 0 ? size : 1));
    if (members == NULL) {
      fprintf(stderr, "out of memory\n");
      free(colors);
      smc_result_free(result);
      return 1;
    }

    if (smc_column_group(result, g, members, size) != 0) {
      fprintf(stderr, "smc_column_group failed for group %d\n", g);
      free(members);
      free(colors);
      smc_result_free(result);
      return 1;
    }

    printf("group %d = {", g);
    for (int k = 0; k < size; k++)
      printf(" %d", members[k]);
    printf(" }\n");

    free(members);
  }

  /* Using the handle after this returns -4 rather than crashing, and freeing
     twice is safe. */
  free(colors);
  smc_result_free(result);

  return 0;
}
