/*
 * test_coloring.c - correctness tests for the colorings produced by libsmc.
 *
 * Nothing is compared against a hard-coded expected coloring: a greedy coloring
 * may legitimately change with the vertex order, so each coloring is verified
 * against the structural property that makes it usable, over all five orders,
 * both postprocessing settings and both dtypes.
 *
 * Compile (after building libsmc with juliac - see interfaces/README.md):
 *   gcc -O2 -o interfaces/build/test_coloring interfaces/test/C/test_coloring.c \
 *       -I interfaces/build/include interfaces/build/lib/libsmc.so \
 *       -Wl,-rpath,'$ORIGIN/lib' -lm
 *
 * Exit code: 0 if all tests pass, 1 otherwise.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smc.h"

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg)                                                  \
  do {                                                                    \
    if (cond) {                                                           \
      n_pass++;                                                           \
    } else {                                                              \
      n_fail++;                                                           \
      printf("  FAIL  %s  (%s:%d)\n", msg, __FILE__, __LINE__);           \
    }                                                                     \
  } while (0)

#define MAXDIM 12
#define MAXNNZ 64

/* A sparsity pattern with values, in CSC form, 0-based. */
typedef struct {
  const char *name;
  int m;
  int n;
  const int *colptr;      /* n + 1 entries */
  const int *rowval;      /* colptr[n] entries */
  const double *nzval;    /* colptr[n] entries */
} Matrix;

/* 4x6 rectangular, nonsymmetric (the `compress` docstring matrix):
 *   . . 4 6 . 9
 *   1 . . . 7 .
 *   . 2 . . 8 .
 *   . 3 5 . . .                                                            */
static const int    ns_colptr[7] = { 0, 1, 3, 5, 6, 8, 9 };
static const int    ns_rowval[9] = { 1, 2, 3, 0, 3, 0, 1, 2, 0 };
static const double ns_nzval[9]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

/* 7x5 rectangular with denser rows and columns. */
static const int    ns2_colptr[6]  = { 0, 4, 7, 10, 13, 16 };
static const int    ns2_rowval[16] = { 0, 1, 4, 6,
                                       1, 2, 5,
                                       2, 3, 6,
                                       0, 3, 4,
                                       1, 4, 5 };
static const double ns2_nzval[16]  = { 1, 3, 2, 7,
                                       4, 6, 5,
                                       7, 8, 8,
                                       2, 9, 3,
                                       5, 4, 6 };

/* 7x7 symmetric, nonzero diagonal: tridiagonal plus the (1,7) corner. */
static const int    sym_colptr[8]  = { 0, 3, 6, 9, 12, 15, 18, 21 };
static const int    sym_rowval[21] = { 0, 1, 6,
                                       0, 1, 2,
                                       1, 2, 3,
                                       2, 3, 4,
                                       3, 4, 5,
                                       4, 5, 6,
                                       0, 5, 6 };
static const double sym_nzval[21]  = { 2, 1, 3,
                                       1, 2, 1,
                                       1, 2, 1,
                                       1, 2, 1,
                                       1, 2, 1,
                                       1, 2, 1,
                                       3, 1, 2 };

/* 6x6 symmetric with a zero diagonal (the 3-cube minus a perfect matching);
   a zero diagonal is what lets postprocessing hand out the neutral color 0. */
static const int    sym0_colptr[7]  = { 0, 2, 4, 6, 8, 10, 12 };
static const int    sym0_rowval[12] = { 1, 2,
                                        0, 3,
                                        0, 4,
                                        1, 5,
                                        2, 5,
                                        3, 4 };
static const double sym0_nzval[12]  = { 1, 2,
                                        1, 3,
                                        2, 4,
                                        3, 5,
                                        4, 6,
                                        5, 6 };

static const Matrix NONSYM_MATRICES[2] = {
  { "nonsym 4x6", 4, 6, ns_colptr, ns_rowval, ns_nzval },
  { "nonsym 7x5", 7, 5, ns2_colptr, ns2_rowval, ns2_nzval }
};

static const Matrix SYM_MATRICES[2] = {
  { "sym 7x7 (full diagonal)", 7, 7, sym_colptr, sym_rowval, sym_nzval },
  { "sym 6x6 (zero diagonal)", 6, 6, sym0_colptr, sym0_rowval, sym0_nzval }
};

/* Dense column-major copy of the matrix. */
static void to_dense(const Matrix *A, double *dense)
{
  int i, j, k;
  for (k = 0; k < A->m * A->n; k++) dense[k] = 0.0;
  for (j = 0; j < A->n; j++)
    for (k = A->colptr[j]; k < A->colptr[j + 1]; k++) {
      i = A->rowval[k];
      dense[j * A->m + i] = A->nzval[k];
    }
}

#define NZ(dense, m, i, j) ((dense)[(j) * (m) + (i)] != 0.0)

/* Two columns of the same nonzero color must have disjoint row supports. */
static int column_colors_are_disjoint(const double *dense, int m, int n, const int *colors)
{
  int i, j, k;
  for (j = 0; j < n; j++)
    for (k = j + 1; k < n; k++) {
      if (colors[j] == 0 || colors[k] == 0) continue;
      if (colors[j] != colors[k]) continue;
      for (i = 0; i < m; i++)
        if (NZ(dense, m, i, j) && NZ(dense, m, i, k)) return 0;
    }
  return 1;
}

/* Same statement on the rows. */
static int row_colors_are_disjoint(const double *dense, int m, int n, const int *colors)
{
  int i, j, k;
  for (i = 0; i < m; i++)
    for (k = i + 1; k < m; k++) {
      if (colors[i] == 0 || colors[k] == 0) continue;
      if (colors[i] != colors[k]) continue;
      for (j = 0; j < n; j++)
        if (NZ(dense, m, i, j) && NZ(dense, m, k, j)) return 0;
    }
  return 1;
}

/* Adjacent vertices carry different nonzero colors. */
static int is_proper(const double *dense, int n, const int *colors)
{
  int i, j;
  for (j = 0; j < n; j++)
    for (i = 0; i < j; i++) {
      if (!NZ(dense, n, i, j)) continue;
      if (colors[i] == 0 || colors[j] == 0) continue;
      if (colors[i] == colors[j]) return 0;
    }
  return 1;
}

/* No path i - j - k - l uses only two colors (star coloring). */
static int is_star_coloring(const double *dense, int n, const int *colors)
{
  int i, j, k, l;
  for (j = 0; j < n; j++)
    for (k = 0; k < n; k++) {
      if (j == k || !NZ(dense, n, j, k)) continue;
      for (i = 0; i < n; i++) {
        if (i == j || i == k || !NZ(dense, n, i, j)) continue;
        for (l = 0; l < n; l++) {
          if (l == i || l == j || l == k || !NZ(dense, n, k, l)) continue;
          if (colors[i] == 0 || colors[j] == 0 || colors[k] == 0 || colors[l] == 0) continue;
          if (colors[i] == colors[k] && colors[j] == colors[l]) return 0;
        }
      }
    }
  return 1;
}

static int uf_find(int *parent, int x)
{
  while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
  return x;
}

/* Every subgraph induced by two colors is a forest (acyclic coloring). */
static int is_acyclic_coloring(const double *dense, int n, const int *colors, int ncolors)
{
  int ca, cb, i, j, parent[MAXDIM];
  for (ca = 1; ca <= ncolors; ca++)
    for (cb = ca + 1; cb <= ncolors; cb++) {
      for (i = 0; i < n; i++) parent[i] = i;
      for (j = 0; j < n; j++)
        for (i = 0; i < j; i++) {
          int ri, rj;
          if (!NZ(dense, n, i, j)) continue;
          if (!((colors[i] == ca && colors[j] == cb) ||
                (colors[i] == cb && colors[j] == ca))) continue;
          ri = uf_find(parent, i);
          rj = uf_find(parent, j);
          if (ri == rj) return 0;        /* a cycle inside {ca, cb} */
          parent[ri] = rj;
        }
    }
  return 1;
}

/* Every nonzero must be readable off the compressed matrix: A[i][j] is either
   the only nonzero of row i among the columns of its color, or the only nonzero
   of column j among the rows of its color.  Pass NULL for the coloring that the
   partition does not produce. */
static int is_directly_recoverable(const double *dense, int m, int n,
                                   const int *row_colors, const int *column_colors)
{
  int i, j, k;
  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      int by_column = 0, by_row = 0;
      if (!NZ(dense, m, i, j)) continue;
      if (column_colors != NULL && column_colors[j] != 0) {
        by_column = 1;
        for (k = 0; k < n; k++)
          if (k != j && NZ(dense, m, i, k) && column_colors[k] == column_colors[j]) by_column = 0;
      }
      if (row_colors != NULL && row_colors[i] != 0) {
        by_row = 1;
        for (k = 0; k < m; k++)
          if (k != i && NZ(dense, m, k, j) && row_colors[k] == row_colors[i]) by_row = 0;
      }
      if (!by_column && !by_row) return 0;
    }
  return 1;
}

/* Symmetric counterpart: A[i][j] is read from B[i][colors[j]] when column j is
   the only one of its color meeting row i, or - by symmetry - from
   B[j][colors[i]]. */
static int is_symmetrically_recoverable(const double *dense, int n, const int *colors)
{
  int i, j, k;
  for (j = 0; j < n; j++)
    for (i = 0; i < n; i++) {
      int by_j = 0, by_i = 0;
      if (!NZ(dense, n, i, j)) continue;
      if (colors[j] != 0) {
        by_j = 1;
        for (k = 0; k < n; k++)
          if (k != j && NZ(dense, n, i, k) && colors[k] == colors[j]) by_j = 0;
      }
      if (colors[i] != 0) {
        by_i = 1;
        for (k = 0; k < n; k++)
          if (k != i && NZ(dense, n, j, k) && colors[k] == colors[i]) by_i = 0;
      }
      if (!by_j && !by_i) return 0;
    }
  return 1;
}

/* The groups are exactly the color classes. */
static int groups_match_colors(void *result, int ngroups, const int *colors, int len,
                               int base, int column)
{
  int g, k, seen[MAXDIM];
  for (k = 0; k < len; k++) seen[k] = 0;

  for (g = 1; g <= ngroups; g++) {
    int size = 0, members[MAXDIM];
    int ret = column ? smc_column_group_size(result, g, &size)
                     : smc_row_group_size(result, g, &size);
    if (ret != 0 || size < 0 || size > len) return 0;
    ret = column ? smc_column_group(result, g, members, size)
                 : smc_row_group(result, g, members, size);
    if (ret != 0) return 0;
    for (k = 0; k < size; k++) {
      int index = members[k] - base;
      if (index < 0 || index >= len) return 0;
      if (seen[index]) return 0;                 /* a member of two groups */
      seen[index] = 1;
      if (colors[index] != g) return 0;          /* wrong group */
    }
  }
  /* Every non-neutral index belongs to exactly one group, and only those. */
  for (k = 0; k < len; k++) {
    if (colors[k] != 0 && !seen[k]) return 0;
    if (colors[k] == 0 && seen[k]) return 0;
    if (colors[k] < 0 || colors[k] > ngroups) return 0;
  }
  return 1;
}

/* The lengths smc_compress and smc_decompress need, read back from the API
   rather than from the Matrix struct, which also checks that smc_nnz and
   smc_size agree with the pattern that was colored.  0 if any query fails. */
static int query_lengths(void *result, const Matrix *A,
                         int *Br_rows, int *Br_cols, int *Bc_rows, int *Bc_cols,
                         size_t *br_len, size_t *bc_len, size_t *a_len, size_t *nzval_len)
{
  int nnz = -1, m = -1, n = -1;

  if (smc_compressed_size(result, Br_rows, Br_cols, Bc_rows, Bc_cols) != 0) return 0;
  if (smc_nnz(result, &nnz) != 0 || nnz != A->colptr[A->n]) return 0;
  if (smc_size(result, &m, &n) != 0 || m != A->m || n != A->n) return 0;

  *br_len = (size_t) *Br_rows * (size_t) *Br_cols;
  *bc_len = (size_t) *Bc_rows * (size_t) *Bc_cols;
  *a_len = (size_t) m * (size_t) n;
  *nzval_len = (size_t) nnz;
  return 1;
}

/* Float64 round trip; returns 1 when every entry of A is reproduced. */
static int roundtrip_f64(void *result, const Matrix *A, int exact)
{
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1, i, j, ok = 1;
  size_t br_len, bc_len, a_len, nzval_len, k;
  double *Br = NULL, *Bc = NULL, *out = NULL, dense[MAXDIM * MAXDIM];

  if (!query_lengths(result, A, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols,
                     &br_len, &bc_len, &a_len, &nzval_len)) return 0;

  if (br_len > 0) Br = (double *) malloc(sizeof(double) * br_len);
  Bc = (double *) malloc(sizeof(double) * bc_len);
  out = (double *) malloc(sizeof(double) * a_len);
  if (Bc == NULL || out == NULL || (br_len > 0 && Br == NULL)) {
    free(Br); free(Bc); free(out); return 0;
  }

  for (k = 0; k < br_len; k++) Br[k] = -987.0;
  for (k = 0; k < bc_len; k++) Bc[k] = -987.0;
  for (k = 0; k < a_len; k++) out[k] = -987.0;

  if (smc_compress(result, A->nzval, nzval_len, Br, br_len, Bc, bc_len) != 0) ok = 0;
  if (smc_decompress(result, Br, br_len, Bc, bc_len, out, a_len) != 0) ok = 0;

  to_dense(A, dense);
  for (j = 0; ok && j < A->n; j++)
    for (i = 0; i < A->m; i++) {
      double expected = dense[j * A->m + i];
      double got = out[j * A->m + i];
      if (exact ? (got != expected) : (fabs(got - expected) > 1e-9 * (1.0 + fabs(expected))))
        ok = 0;
    }

  free(Br); free(Bc); free(out);
  return ok;
}

/* Float32 round trip on a handle created with dtype == SMC_FLOAT32. */
static int roundtrip_f32(void *result, const Matrix *A, int exact)
{
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1, i, j, ok = 1;
  size_t br_len, bc_len, a_len, nzval_len, k;
  float *Br = NULL, *Bc = NULL, *out = NULL, nzval32[MAXNNZ];
  double dense[MAXDIM * MAXDIM];

  if (!query_lengths(result, A, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols,
                     &br_len, &bc_len, &a_len, &nzval_len)) return 0;
  for (k = 0; k < nzval_len; k++) nzval32[k] = (float) A->nzval[k];

  if (br_len > 0) Br = (float *) malloc(sizeof(float) * br_len);
  Bc = (float *) malloc(sizeof(float) * bc_len);
  out = (float *) malloc(sizeof(float) * a_len);
  if (Bc == NULL || out == NULL || (br_len > 0 && Br == NULL)) {
    free(Br); free(Bc); free(out); return 0;
  }

  for (k = 0; k < br_len; k++) Br[k] = -987.0f;
  for (k = 0; k < bc_len; k++) Bc[k] = -987.0f;
  for (k = 0; k < a_len; k++) out[k] = -987.0f;

  /* Lengths are element counts, so they match the Float64 numbers. */
  if (smc_compress(result, nzval32, nzval_len, Br, br_len, Bc, bc_len) != 0) ok = 0;
  if (smc_decompress(result, Br, br_len, Bc, bc_len, out, a_len) != 0) ok = 0;

  to_dense(A, dense);
  for (j = 0; ok && j < A->n; j++)
    for (i = 0; i < A->m; i++) {
      float expected = (float) dense[j * A->m + i];
      float got = out[j * A->m + i];
      if (exact ? (got != expected) : (fabsf(got - expected) > 1e-4f * (1.0f + fabsf(expected))))
        ok = 0;
    }

  free(Br); free(Bc); free(out);
  return ok;
}

/* Every buffer length, one element short, one at a time.  Each buffer keeps its
   full size and is pre-filled, so a -3 can only come from the length check and a
   surviving sentinel proves the check ran before any element was written. */
static int length_guards_ok(void *result, const Matrix *A)
{
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1, ok = 1;
  int bidirectional;
  size_t br_len, bc_len, a_len, nzval_len, k;
  double *Br = NULL, *Bc = NULL, *out = NULL;

  if (!query_lengths(result, A, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols,
                     &br_len, &bc_len, &a_len, &nzval_len)) return 0;
  /* Br_cols identifies the partition; br_len does not, since postprocessing can
     leave zero row groups and make Br 0-by-n. */
  bidirectional = (Br_cols > 0);

  if (bidirectional) Br = (double *) malloc(sizeof(double) * (br_len ? br_len : 1));
  Bc = (double *) malloc(sizeof(double) * (bc_len ? bc_len : 1));
  out = (double *) malloc(sizeof(double) * (a_len ? a_len : 1));
  if (Bc == NULL || out == NULL || (bidirectional && Br == NULL)) {
    free(Br); free(Bc); free(out); return 0;
  }

  /* Exact sizes work, and fill Bc/Br with something a decompress can use. */
  if (smc_compress(result, A->nzval, nzval_len, Br, br_len, Bc, bc_len) != 0) ok = 0;
  if (smc_decompress(result, Br, br_len, Bc, bc_len, out, a_len) != 0) ok = 0;

  /* Only understate a length that is positive: `len - 1` on a size_t 0 wraps to
     SIZE_MAX, a valid enormous promise that is rightly accepted.  A length can
     be 0 -- postprocessing can leave zero column groups. */
  if (nzval_len > 0 &&
      smc_compress(result, A->nzval, nzval_len - 1, Br, br_len, Bc, bc_len) != -3) ok = 0;
  if (bc_len > 0 &&
      smc_compress(result, A->nzval, nzval_len, Br, br_len, Bc, bc_len - 1) != -3) ok = 0;
  if (smc_compress(result, NULL, nzval_len, Br, br_len, Bc, bc_len) != -3) ok = 0;
  if (bidirectional) {
    if (br_len > 0 &&
        smc_compress(result, A->nzval, nzval_len, Br, br_len - 1, Bc, bc_len) != -3) ok = 0;
    /* Both compressed matrices are required. */
    if (smc_compress(result, A->nzval, nzval_len, NULL, 0, Bc, bc_len) != -3) ok = 0;
  } else {
    /* Br is unused: NULL with a length of 0 is the documented call. */
    if (smc_compress(result, A->nzval, nzval_len, NULL, 0, Bc, bc_len) != 0) ok = 0;
  }

  for (k = 0; k < a_len; k++) out[k] = -987.0;
  if (a_len > 0 &&
      smc_decompress(result, Br, br_len, Bc, bc_len, out, a_len - 1) != -3) ok = 0;
  if (bc_len > 0 &&
      smc_decompress(result, Br, br_len, Bc, bc_len - 1, out, a_len) != -3) ok = 0;
  if (bidirectional && br_len > 0 &&
      smc_decompress(result, Br, br_len - 1, Bc, bc_len, out, a_len) != -3)
    ok = 0;
  if (bidirectional && smc_decompress(result, NULL, 0, Bc, bc_len, out, a_len) != -3) ok = 0;
  if (smc_decompress(result, Br, br_len, Bc, bc_len, NULL, a_len) != -3) ok = 0;
  for (k = 0; k < a_len; k++) if (out[k] != -987.0) ok = 0;

  if (smc_decompress(result, Br, br_len, Bc, bc_len, out, a_len) != 0) ok = 0;

  free(Br); free(Bc); free(out);
  return ok;
}

static void test_case(const Matrix *A, SmcColoringOptions o, const char *label)
{
  void *result = NULL;
  int rows[MAXDIM] = { 0 }, columns[MAXDIM] = { 0 };
  int ncolors = 0, nrow_groups = 0, ncolumn_groups = 0;
  int has_columns = (o.partition != SMC_ROW);
  int has_rows = (o.partition != SMC_COLUMN);
  int direct = (o.decompression == SMC_DIRECT);
  double dense[MAXDIM * MAXDIM];

  if (smc_coloring(A->m, A->n, A->colptr, A->rowval, &o, &result) != 0) {
    n_fail++;
    printf("  FAIL  smc_coloring failed for %s / %s\n", A->name, label);
    return;
  }

  to_dense(A, dense);
  CHECK(smc_ncolors(result, &ncolors) == 0 && ncolors > 0, "ncolors is positive");

  if (has_columns) {
    CHECK(smc_column_colors(result, columns, A->n) == 0, "column colors");
    CHECK(smc_ncolumn_groups(result, &ncolumn_groups) == 0, "ncolumn_groups");
    CHECK(groups_match_colors(result, ncolumn_groups, columns, A->n, o.index_base, 1),
          "column groups are exactly the color classes");
  }
  if (has_rows) {
    CHECK(smc_row_colors(result, rows, A->m) == 0, "row colors");
    CHECK(smc_nrow_groups(result, &nrow_groups) == 0, "nrow_groups");
    CHECK(groups_match_colors(result, nrow_groups, rows, A->m, o.index_base, 0),
          "row groups are exactly the color classes");
  }

  if (o.partition == SMC_BIDIRECTIONAL)
    CHECK(ncolors == nrow_groups + ncolumn_groups, "ncolors counts both dimensions");
  else if (has_columns)
    CHECK(ncolors == ncolumn_groups, "ncolors is the number of column groups");
  else
    CHECK(ncolors == nrow_groups, "ncolors is the number of row groups");

  /* ---- structural validity, from the pattern alone ---------------------- */
  if (o.structure == SMC_NONSYMMETRIC && o.partition == SMC_COLUMN) {
    CHECK(column_colors_are_disjoint(dense, A->m, A->n, columns),
          "columns of one color share no nonzero row");
    CHECK(is_directly_recoverable(dense, A->m, A->n, NULL, columns),
          "every nonzero is recoverable from its column group");
  } else if (o.structure == SMC_NONSYMMETRIC && o.partition == SMC_ROW) {
    CHECK(row_colors_are_disjoint(dense, A->m, A->n, rows),
          "rows of one color share no nonzero column");
    CHECK(is_directly_recoverable(dense, A->m, A->n, rows, NULL),
          "every nonzero is recoverable from its row group");
  } else if (o.structure == SMC_SYMMETRIC) {
    CHECK(is_proper(dense, A->n, columns), "the symmetric coloring is proper");
    if (direct) {
      CHECK(is_star_coloring(dense, A->n, columns), "the symmetric coloring is a star coloring");
      CHECK(is_symmetrically_recoverable(dense, A->n, columns),
            "every nonzero is recoverable, using the symmetry of the matrix");
    } else {
      CHECK(is_acyclic_coloring(dense, A->n, columns, ncolors),
            "the symmetric coloring is acyclic");
    }
  } else if (direct) {                 /* nonsymmetric bidirectional, direct */
    CHECK(is_directly_recoverable(dense, A->m, A->n, rows, columns),
          "every nonzero is recoverable from a row or a column group");
  }

  /* ---- the compressed form determines the matrix ------------------------ */
  if (o.dtype == SMC_FLOAT32) {
    CHECK(roundtrip_f32(result, A, direct), "Float32 compress -> decompress round trip");
  } else {
    CHECK(roundtrip_f64(result, A, direct), "Float64 compress -> decompress round trip");
    /* The length checks count elements, so one Float64 pass covers them. */
    CHECK(length_guards_ok(result, A), "every buffer length is checked, and short is -3");
  }

  CHECK(smc_result_free(result) == 0, "free");
}

static void test_fast_coloring(const Matrix *A, SmcColoringOptions o)
{
  void *result = NULL;
  int rows[MAXDIM], columns[MAXDIM];
  int frows[MAXDIM], fcolumns[MAXDIM];
  int ncolors = 0, fncolors = -1, k, same = 1;

  if (smc_coloring(A->m, A->n, A->colptr, A->rowval, &o, &result) != 0) {
    n_fail++;
    printf("  FAIL  smc_coloring failed in the fast_coloring comparison\n");
    return;
  }
  smc_ncolors(result, &ncolors);
  if (o.partition != SMC_ROW) smc_column_colors(result, columns, A->n);
  if (o.partition != SMC_COLUMN) smc_row_colors(result, rows, A->m);
  smc_result_free(result);

  CHECK(smc_fast_coloring(A->m, A->n, A->colptr, A->rowval, &o,
                          frows, fcolumns, &fncolors) == 0, "smc_fast_coloring succeeds");
  CHECK(fncolors == ncolors, "smc_fast_coloring reports the same number of colors");
  if (o.partition != SMC_ROW)
    for (k = 0; k < A->n; k++) if (fcolumns[k] != columns[k]) same = 0;
  if (o.partition != SMC_COLUMN)
    for (k = 0; k < A->m; k++) if (frows[k] != rows[k]) same = 0;
  CHECK(same, "smc_fast_coloring agrees with smc_coloring");

  /* A buffer may be NULL exactly when the partition produces no coloring for
     that dimension. */
  fncolors = -1;
  CHECK(smc_fast_coloring(A->m, A->n, A->colptr, A->rowval, &o, NULL, fcolumns, &fncolors) ==
        (o.partition == SMC_COLUMN ? 0 : -3), "NULL row_colors");
  fncolors = -1;
  CHECK(smc_fast_coloring(A->m, A->n, A->colptr, A->rowval, &o, frows, NULL, &fncolors) ==
        (o.partition == SMC_ROW ? 0 : -3), "NULL column_colors");
}

static const int SUPPORTED[6][3] = {
  { SMC_NONSYMMETRIC, SMC_COLUMN,        SMC_DIRECT       },
  { SMC_NONSYMMETRIC, SMC_ROW,           SMC_DIRECT       },
  { SMC_SYMMETRIC,    SMC_COLUMN,        SMC_DIRECT       },
  { SMC_SYMMETRIC,    SMC_COLUMN,        SMC_SUBSTITUTION },
  { SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT       },
  { SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION }
};

static const char *STRUCTURE_NAME[2] = { "nonsymmetric", "symmetric" };
static const char *PARTITION_NAME[3] = { "column", "row", "bidirectional" };
static const char *DECOMPRESSION_NAME[2] = { "direct", "substitution" };
static const char *ORDER_NAME[5] = { "natural", "largest_first", "smallest_last",
                                     "incidence_degree", "dynamic_largest_first" };

static void test_all_combinations(void)
{
  int c, order, post, dt, k;
  char label[128];

  for (c = 0; c < 6; c++) {
    const Matrix *matrices = (SUPPORTED[c][0] == SMC_SYMMETRIC) ? SYM_MATRICES : NONSYM_MATRICES;
    printf("%s / %s / %s ...\n", STRUCTURE_NAME[SUPPORTED[c][0]],
           PARTITION_NAME[SUPPORTED[c][1]], DECOMPRESSION_NAME[SUPPORTED[c][2]]);
    for (order = 0; order < 5; order++)
      for (post = 0; post < 2; post++)
        for (dt = 0; dt < 2; dt++)
          for (k = 0; k < 2; k++) {
            SmcColoringOptions o = smc_default_options();
            o.structure = SUPPORTED[c][0];
            o.partition = SUPPORTED[c][1];
            o.decompression = SUPPORTED[c][2];
            o.order = order;
            o.postprocessing = post;
            o.dtype = dt;
            snprintf(label, sizeof(label), "order=%s postprocessing=%d dtype=%d",
                     ORDER_NAME[order], post, dt);
            test_case(&matrices[k], o, label);
          }
    /* fast_coloring only needs one pass per combination. */
    for (k = 0; k < 2; k++) {
      SmcColoringOptions o = smc_default_options();
      o.structure = SUPPORTED[c][0];
      o.partition = SUPPORTED[c][1];
      o.decompression = SUPPORTED[c][2];
      o.order = SMC_LARGEST_FIRST;
      test_fast_coloring(&matrices[k], o);
    }
  }
}

/* index_base shifts the pattern indices, never the colors. */
static void test_index_base(void)
{
  int c, k;

  printf("index_base 0 vs 1 ...\n");
  for (c = 0; c < 6; c++) {
    const Matrix *matrices = (SUPPORTED[c][0] == SMC_SYMMETRIC) ? SYM_MATRICES : NONSYM_MATRICES;
    for (k = 0; k < 2; k++) {
      const Matrix *A = &matrices[k];
      SmcColoringOptions o0 = smc_default_options();
      SmcColoringOptions o1 = smc_default_options();
      void *r0 = NULL, *r1 = NULL;
      int colptr1[MAXDIM + 1], rowval1[MAXNNZ], j, same = 1;
      int c0[MAXDIM], c1[MAXDIM], nc0 = 0, nc1 = 0;

      o0.structure = o1.structure = SUPPORTED[c][0];
      o0.partition = o1.partition = SUPPORTED[c][1];
      o0.decompression = o1.decompression = SUPPORTED[c][2];
      o1.index_base = 1;

      for (j = 0; j <= A->n; j++) colptr1[j] = A->colptr[j] + 1;
      for (j = 0; j < A->colptr[A->n]; j++) rowval1[j] = A->rowval[j] + 1;

      CHECK(smc_coloring(A->m, A->n, A->colptr, A->rowval, &o0, &r0) == 0, "0-based coloring");
      CHECK(smc_coloring(A->m, A->n, colptr1, rowval1, &o1, &r1) == 0, "1-based coloring");
      smc_ncolors(r0, &nc0);
      smc_ncolors(r1, &nc1);
      CHECK(nc0 == nc1, "index_base does not change the number of colors");

      if (o0.partition != SMC_ROW) {
        smc_column_colors(r0, c0, A->n);
        smc_column_colors(r1, c1, A->n);
        for (j = 0; j < A->n; j++) if (c0[j] != c1[j]) same = 0;
      }
      if (o0.partition != SMC_COLUMN) {
        smc_row_colors(r0, c0, A->m);
        smc_row_colors(r1, c1, A->m);
        for (j = 0; j < A->m; j++) if (c0[j] != c1[j]) same = 0;
      }
      CHECK(same, "index_base does not change the colors");

      smc_result_free(r0);
      smc_result_free(r1);
    }
  }
}

/* postprocessing may only replace colors by the neutral color 0, never make the
   coloring worse. */
static void test_postprocessing(void)
{
  int k;

  printf("postprocessing ...\n");
  for (k = 0; k < 2; k++) {
    const Matrix *A = &SYM_MATRICES[k];
    SmcColoringOptions off = smc_default_options();
    SmcColoringOptions on = smc_default_options();
    void *r_off = NULL, *r_on = NULL;
    int c_off[MAXDIM], c_on[MAXDIM], nc_off = 0, nc_on = 0, j, l, valid = 1, injective = 1;

    off.structure = on.structure = SMC_SYMMETRIC;
    on.postprocessing = 1;

    CHECK(smc_coloring(A->m, A->n, A->colptr, A->rowval, &off, &r_off) == 0, "coloring");
    CHECK(smc_coloring(A->m, A->n, A->colptr, A->rowval, &on, &r_on) == 0, "coloring (postprocessed)");
    smc_ncolors(r_off, &nc_off);
    smc_ncolors(r_on, &nc_on);
    smc_column_colors(r_off, c_off, A->n);
    smc_column_colors(r_on, c_on, A->n);

    CHECK(nc_on <= nc_off, "postprocessing never increases the number of colors");
    for (j = 0; j < A->n; j++) {
      if (c_on[j] < 0 || c_on[j] > nc_on) valid = 0;
      if (c_off[j] < 1 || c_off[j] > nc_off) valid = 0;
    }
    CHECK(valid, "colors stay within 0..ncolors, and are nonzero without postprocessing");

    /* Surviving colors are renamed injectively: two vertices keep the same
       nonzero color together. */
    for (j = 0; j < A->n; j++)
      for (l = j + 1; l < A->n; l++) {
        if (c_on[j] == 0 || c_on[l] == 0) continue;
        if ((c_on[j] == c_on[l]) != (c_off[j] == c_off[l])) injective = 0;
      }
    CHECK(injective, "postprocessing renames colors injectively");

    smc_result_free(r_off);
    smc_result_free(r_on);
  }
}

int main(void)
{
  test_all_combinations();
  test_index_base();
  test_postprocessing();

  printf("\n%d checks passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
