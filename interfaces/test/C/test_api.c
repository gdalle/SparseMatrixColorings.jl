/*
 * test_api.c - ABI and error-path tests for the libsmc C interface.
 *
 * Covers the struct layout and enum values (a drift between smc.h and
 * c_enums.jl is silent memory corruption otherwise), the return codes -2/-3/-4,
 * the sizing queries, and every buffer length of compress/decompress.
 * test_coloring.c checks that the colorings themselves are correct.
 *
 * Compile (after building libsmc with juliac - see interfaces/README.md):
 *   gcc -O2 -o interfaces/build/test_api interfaces/test/C/test_api.c \
 *       -I interfaces/build/include interfaces/build/lib/libsmc.so \
 *       -Wl,-rpath,'$ORIGIN/lib' -lm
 *
 * Exit code: 0 if all tests pass, 1 otherwise.
 */

#include <stddef.h>
#include <stdint.h>
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

/* Compile-time assertion that does not require C11. */
#define SMC_STATIC_ASSERT(cond, name) \
  typedef char smc_static_assert_##name[(cond) ? 1 : -1]

/* The layout of SmcColoringOptions is part of the contract: checked at compile
   time (a mismatched header fails the build) and again at run time. */

SMC_STATIC_ASSERT(sizeof(SmcColoringOptions) == 8 * sizeof(int), options_size);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, structure) == 0 * sizeof(int), f0);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, partition) == 1 * sizeof(int), f1);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, decompression) == 2 * sizeof(int), f2);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, order) == 3 * sizeof(int), f3);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, postprocessing) == 4 * sizeof(int), f4);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, symmetric_pattern) == 5 * sizeof(int), f5);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, index_base) == 6 * sizeof(int), f6);
SMC_STATIC_ASSERT(offsetof(SmcColoringOptions, dtype) == 7 * sizeof(int), f7);

SMC_STATIC_ASSERT(SMC_FLOAT64 == 0 && SMC_FLOAT32 == 1, dtype_values);
SMC_STATIC_ASSERT(SMC_NONSYMMETRIC == 0 && SMC_SYMMETRIC == 1, structure_values);
SMC_STATIC_ASSERT(SMC_COLUMN == 0 && SMC_ROW == 1 && SMC_BIDIRECTIONAL == 2, partition_values);
SMC_STATIC_ASSERT(SMC_DIRECT == 0 && SMC_SUBSTITUTION == 1, decompression_values);
SMC_STATIC_ASSERT(SMC_NATURAL == 0 && SMC_LARGEST_FIRST == 1 && SMC_SMALLEST_LAST == 2 &&
                  SMC_INCIDENCE_DEGREE == 3 && SMC_DYNAMIC_LARGEST_FIRST == 4, order_values);

/* The 4x6 pattern of the `compress` docstring, 0-based and 1-based:
 *   . . 4 6 . 9
 *   1 . . . 7 .
 *   . 2 . . 8 .
 *   . 3 5 . . .                                                             */

#define M 4
#define N 6
#define NNZ 9

static const int colptr0[N + 1] = { 0, 1, 3, 5, 6, 8, 9 };
static const int rowval0[NNZ]   = { 1, 2, 3, 0, 3, 0, 1, 2, 0 };
static const double nzval[NNZ]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

static int colptr1[N + 1];
static int rowval1[NNZ];

static void build_one_based(void)
{
  int j;
  for (j = 0; j <= N; j++) colptr1[j] = colptr0[j] + 1;
  for (j = 0; j < NNZ; j++) rowval1[j] = rowval0[j] + 1;
}

/* A 5x5 symmetric pattern with a nonzero diagonal, for the symmetric paths. */
#define SM 5
#define SNNZ 13
static const int scolptr[SM + 1] = { 0, 3, 6, 8, 11, 13 };
static const int srowval[SNNZ]   = { 0, 1, 3,
                                     0, 1, 2,
                                     1, 2,
                                     0, 3, 4,
                                     3, 4 };
/* The six supported (structure, partition, decompression) triples. */
static const int SUPPORTED[6][3] = {
  { SMC_NONSYMMETRIC, SMC_COLUMN,        SMC_DIRECT       },
  { SMC_NONSYMMETRIC, SMC_ROW,           SMC_DIRECT       },
  { SMC_SYMMETRIC,    SMC_COLUMN,        SMC_DIRECT       },
  { SMC_SYMMETRIC,    SMC_COLUMN,        SMC_SUBSTITUTION },
  { SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT       },
  { SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION }
};

/* The sentinel that a rejected call must leave in place. */
#define SENTINEL (-987.0)

static void fill_sentinel(double *p, size_t len)
{
  size_t i;
  for (i = 0; i < len; i++) p[i] = SENTINEL;
}

static int all_sentinel(const double *p, size_t len)
{
  size_t i;
  for (i = 0; i < len; i++) if (p[i] != SENTINEL) return 0;
  return 1;
}

static void test_abi(void)
{
  SmcColoringOptions o;
  const int *base = (const int *) &o;

  printf("ABI (struct layout and enum values) ...\n");
  CHECK(sizeof(SmcColoringOptions) == 8 * sizeof(int),
        "sizeof(SmcColoringOptions) == 8 * sizeof(int)");

  /* Eight consecutive ints, no padding. */
  o.structure = 10; o.partition = 11; o.decompression = 12; o.order = 13;
  o.postprocessing = 14; o.symmetric_pattern = 15; o.index_base = 16; o.dtype = 17;
  CHECK(base[0] == 10 && base[1] == 11 && base[2] == 12 && base[3] == 13 &&
        base[4] == 14 && base[5] == 15 && base[6] == 16 && base[7] == 17,
        "SmcColoringOptions is eight consecutive ints, in the documented order");

  CHECK(SMC_FLOAT64 == 0 && SMC_FLOAT32 == 1, "SmcDataType values");
  CHECK(SMC_NONSYMMETRIC == 0 && SMC_SYMMETRIC == 1, "SmcStructure values");
  CHECK(SMC_COLUMN == 0 && SMC_ROW == 1 && SMC_BIDIRECTIONAL == 2, "SmcPartition values");
  CHECK(SMC_DIRECT == 0 && SMC_SUBSTITUTION == 1, "SmcDecompression values");
  CHECK(SMC_NATURAL == 0 && SMC_LARGEST_FIRST == 1 && SMC_SMALLEST_LAST == 2 &&
        SMC_INCIDENCE_DEGREE == 3 && SMC_DYNAMIC_LARGEST_FIRST == 4, "SmcOrder values");
}

static void test_default_options(void)
{
  SmcColoringOptions o = smc_default_options();

  printf("default options ...\n");
  CHECK(o.structure == SMC_NONSYMMETRIC, "default structure is SMC_NONSYMMETRIC");
  CHECK(o.partition == SMC_COLUMN, "default partition is SMC_COLUMN");
  CHECK(o.decompression == SMC_DIRECT, "default decompression is SMC_DIRECT");
  CHECK(o.order == SMC_NATURAL, "default order is SMC_NATURAL");
  CHECK(o.postprocessing == 0, "default postprocessing is 0");
  CHECK(o.symmetric_pattern == 0, "default symmetric_pattern is 0");
  CHECK(o.index_base == 0, "default index_base is 0");
  CHECK(o.dtype == SMC_FLOAT64, "default dtype is SMC_FLOAT64");
}

static void test_version(void)
{
  int major = -1, minor = -1, patch = -1;

  printf("version ...\n");
  smc_version(&major, &minor, &patch);
  CHECK(major == SMC_VERSION_MAJOR && minor == SMC_VERSION_MINOR &&
        patch == SMC_VERSION_PATCH,
        "smc_version matches the SMC_VERSION_* macros");
}

/* A NULL options pointer must behave exactly like smc_default_options(). */
static void test_null_options(void)
{
  void *with_null = NULL, *with_defaults = NULL;
  SmcColoringOptions o = smc_default_options();
  int a[N], b[N], j, same = 1;

  printf("NULL options ...\n");
  CHECK(smc_coloring(M, N, colptr0, rowval0, NULL, &with_null) == 0,
        "smc_coloring accepts NULL options");
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &with_defaults) == 0,
        "smc_coloring accepts smc_default_options()");
  CHECK(smc_column_colors(with_null, a, N) == 0, "colors with NULL options");
  CHECK(smc_column_colors(with_defaults, b, N) == 0, "colors with explicit defaults");
  for (j = 0; j < N; j++) if (a[j] != b[j]) same = 0;
  CHECK(same, "NULL options == smc_default_options()");
  smc_result_free(with_null);
  smc_result_free(with_defaults);
}

static void test_unsupported_combinations(void)
{
  /* The six triples that are not a SparseMatrixColorings problem; see smc.h. */
  static const int combos[6][3] = {
    { SMC_NONSYMMETRIC, SMC_COLUMN,        SMC_SUBSTITUTION },
    { SMC_NONSYMMETRIC, SMC_ROW,           SMC_SUBSTITUTION },
    { SMC_SYMMETRIC,    SMC_ROW,           SMC_DIRECT       },
    { SMC_SYMMETRIC,    SMC_ROW,           SMC_SUBSTITUTION },
    { SMC_SYMMETRIC,    SMC_BIDIRECTIONAL, SMC_DIRECT       },
    { SMC_SYMMETRIC,    SMC_BIDIRECTIONAL, SMC_SUBSTITUTION }
  };
  int k, dt;

  printf("unsupported combinations ...\n");
  for (k = 0; k < 6; k++) {
    for (dt = 0; dt < 2; dt++) {
      SmcColoringOptions o = smc_default_options();
      void *result = (void *) 0x1;          /* poison: must be left untouched */
      int rows[SM], cols[SM], nc = -1, ret;

      o.structure = combos[k][0];
      o.partition = combos[k][1];
      o.decompression = combos[k][2];
      o.dtype = dt;

      ret = smc_coloring(SM, SM, scolptr, srowval, &o, &result);
      CHECK(ret == -2, "unsupported combination returns -2");

      ret = smc_fast_coloring(SM, SM, scolptr, srowval, &o, rows, cols, &nc);
      CHECK(ret == -2, "smc_fast_coloring rejects the same combination with -2");
    }
  }
}

static void test_invalid_arguments(void)
{
  SmcColoringOptions o = smc_default_options();
  void *result = NULL;
  int colors[N];

  printf("invalid arguments ...\n");
  CHECK(smc_coloring(M, N, NULL, rowval0, &o, &result) == -3, "NULL colptr returns -3");
  CHECK(smc_coloring(M, N, colptr0, NULL, &o, &result) == -3, "NULL rowval returns -3");
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, NULL) == -3, "NULL result_out returns -3");
  CHECK(smc_coloring(0, N, colptr0, rowval0, &o, &result) == -3, "m == 0 returns -3");
  CHECK(smc_coloring(M, 0, colptr0, rowval0, &o, &result) == -3, "n == 0 returns -3");
  CHECK(smc_coloring(-1, N, colptr0, rowval0, &o, &result) == -3, "m < 0 returns -3");
  CHECK(result == NULL, "the handle is left untouched when the call fails");

  {
    int f;
    for (f = 0; f < 6; f++) {
      SmcColoringOptions bad = smc_default_options();
      switch (f) {
        case 0: bad.structure = 5; break;
        case 1: bad.partition = 9; break;
        case 2: bad.decompression = 7; break;
        case 3: bad.order = 9; break;
        case 4: bad.dtype = 4; break;
        default: bad.index_base = 2; break;
      }
      result = NULL;
      CHECK(smc_coloring(M, N, colptr0, rowval0, &bad, &result) == -3,
            "out-of-range enum or index_base returns -3");
      CHECK(result == NULL, "no handle is created for a rejected option");
    }
  }

  result = NULL;
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &result) == 0, "reference coloring");
  CHECK(smc_column_colors(result, colors, N - 1) == -3, "len < n returns -3");
  CHECK(smc_column_colors(result, NULL, N) == -3, "NULL colors buffer returns -3");
  CHECK(smc_ncolors(result, NULL) == -3, "NULL ncolors_out returns -3");
  CHECK(smc_ncolumn_groups(result, NULL) == -3, "NULL ngroups_out returns -3");
  CHECK(smc_compressed_size(result, NULL, NULL, NULL, NULL) == -3,
        "NULL size outputs return -3");
  {
    int ngroups = 0, size = 0, members[N];
    CHECK(smc_ncolumn_groups(result, &ngroups) == 0, "ncolumn_groups");
    CHECK(smc_column_group_size(result, 0, &size) == -3, "group 0 is out of range");
    CHECK(smc_column_group_size(result, ngroups + 1, &size) == -3,
          "group ngroups+1 is out of range");
    CHECK(smc_column_group_size(result, 1, &size) == 0 && size > 0, "group 1 has a size");
    CHECK(smc_column_group(result, 1, members, size - 1) == -3, "short group buffer returns -3");
    CHECK(smc_column_group(result, 1, NULL, size) == -3, "NULL group buffer returns -3");
  }
  {
    int rows[M], nrow_groups = 0;
    CHECK(smc_row_colors(result, rows, M) == -2, "smc_row_colors on a column partition returns -2");
    CHECK(smc_nrow_groups(result, &nrow_groups) == -2, "smc_nrow_groups on a column partition returns -2");
  }
  smc_result_free(result);
}

static void test_invalid_handle(void)
{
  SmcColoringOptions o = smc_default_options();
  void *result = NULL, *bogus = (void *) (uintptr_t) 0xdeadbeef0ULL;
  int colors[N], value = 0, members[N], null_ret;
  double Bc[M * N];

  printf("invalid and already-freed handles ...\n");
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &result) == 0, "coloring for the free test");
  CHECK(smc_result_free(result) == 0, "first free returns 0");
  CHECK(smc_result_free(result) == -4, "double free returns -4");

  /* Every entry point must reject the stale handle rather than dereference it. */
  CHECK(smc_ncolors(result, &value) == -4, "smc_ncolors after free returns -4");
  CHECK(smc_column_colors(result, colors, N) == -4, "smc_column_colors after free returns -4");
  CHECK(smc_row_colors(result, colors, M) == -4, "smc_row_colors after free returns -4");
  CHECK(smc_ncolumn_groups(result, &value) == -4, "smc_ncolumn_groups after free returns -4");
  CHECK(smc_nrow_groups(result, &value) == -4, "smc_nrow_groups after free returns -4");
  CHECK(smc_column_group_size(result, 1, &value) == -4, "smc_column_group_size after free returns -4");
  CHECK(smc_column_group(result, 1, members, N) == -4, "smc_column_group after free returns -4");
  CHECK(smc_row_group_size(result, 1, &value) == -4, "smc_row_group_size after free returns -4");
  CHECK(smc_row_group(result, 1, members, M) == -4, "smc_row_group after free returns -4");
  CHECK(smc_compressed_size(result, &value, &value, &value, &value) == -4,
        "smc_compressed_size after free returns -4");
  CHECK(smc_nnz(result, &value) == -4, "smc_nnz after free returns -4");
  CHECK(smc_size(result, &value, &value) == -4, "smc_size after free returns -4");
  CHECK(smc_compress(result, nzval, (size_t) NNZ, NULL, 0, Bc, (size_t) (M * N)) == -4,
        "smc_compress after free returns -4");
  CHECK(smc_decompress(result, NULL, 0, Bc, (size_t) (M * N), Bc, (size_t) (M * N)) == -4,
        "smc_decompress after free returns -4");

  CHECK(smc_result_free(bogus) == -4, "freeing a never-allocated handle returns -4");
  CHECK(smc_ncolors(bogus, &value) == -4, "querying a never-allocated handle returns -4");
  null_ret = smc_result_free(NULL);
  CHECK(null_ret == -3 || null_ret == -4, "freeing NULL is rejected, not a crash");
}

/* smc_nnz and smc_size are the only way a caller holding nothing but a handle
   can size nzval and A_out. */

static void test_sizing_queries(void)
{
  int k, dt;

  printf("smc_nnz / smc_size ...\n");
  for (k = 0; k < 6; k++) {
    for (dt = 0; dt < 2; dt++) {
      SmcColoringOptions o = smc_default_options();
      void *result = NULL;
      int symmetric = (SUPPORTED[k][0] == SMC_SYMMETRIC);
      int m = symmetric ? SM : M;
      int n = symmetric ? SM : N;
      int want_nnz = symmetric ? SNNZ : NNZ;
      const int *cp = symmetric ? scolptr : colptr0;
      const int *rv = symmetric ? srowval : rowval0;
      int got_nnz = -1, got_m = -1, got_n = -1;

      o.structure = SUPPORTED[k][0];
      o.partition = SUPPORTED[k][1];
      o.decompression = SUPPORTED[k][2];
      o.dtype = dt;

      CHECK(smc_coloring(m, n, cp, rv, &o, &result) == 0, "coloring for the sizing queries");
      CHECK(smc_nnz(result, &got_nnz) == 0 && got_nnz == want_nnz,
            "smc_nnz is the number of stored entries");
      CHECK(smc_size(result, &got_m, &got_n) == 0 && got_m == m && got_n == n,
            "smc_size is the shape of the colored matrix");

      CHECK(smc_nnz(result, NULL) == -3, "NULL nnz_out returns -3");
      CHECK(smc_size(result, NULL, &got_n) == -3, "NULL m_out returns -3");
      CHECK(smc_size(result, &got_m, NULL) == -3, "NULL n_out returns -3");

      CHECK(smc_result_free(result) == 0, "free");
      CHECK(smc_nnz(result, &got_nnz) == -4, "smc_nnz on a freed handle returns -4");
      CHECK(smc_size(result, &got_m, &got_n) == -4, "smc_size on a freed handle returns -4");
    }
  }
}

/* Buffer lengths are element counts, checked before any element is read or
 * written.  Each length is understated on its own with the buffer left at full
 * size, so a -3 can only come from the length check and a surviving sentinel
 * proves the check ran before the write.
 *
 * Column partition: Br is unused, so NULL with a length of 0 is the documented
 * call and only nzval_len, Bc_len and A_len are in play. */
static void test_buffer_lengths(void)
{
  SmcColoringOptions o = smc_default_options();
  void *result = NULL;
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1;
  int nnz = -1, m = -1, n = -1;
  size_t bc_len, a_len;
  double *Bc, *A;

  printf("buffer lengths (column partition) ...\n");
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &result) == 0, "coloring");
  CHECK(smc_nnz(result, &nnz) == 0 && nnz == NNZ, "nnz");
  CHECK(smc_size(result, &m, &n) == 0 && m == M && n == N, "size");
  CHECK(smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols) == 0,
        "compressed size");
  CHECK(Br_rows == 0 && Br_cols == 0, "a column partition has no Br");

  bc_len = (size_t) Bc_rows * (size_t) Bc_cols;
  a_len = (size_t) m * (size_t) n;
  Bc = (double *) malloc(sizeof(double) * bc_len);
  A = (double *) malloc(sizeof(double) * a_len);
  CHECK(Bc != NULL && A != NULL, "allocation");
  if (Bc == NULL || A == NULL) { free(Bc); free(A); smc_result_free(result); return; }

  CHECK(smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, bc_len) == 0,
        "exact-size compress succeeds");
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, a_len) == 0,
        "exact-size decompress succeeds");

  fill_sentinel(Bc, bc_len);
  CHECK(smc_compress(result, nzval, (size_t) nnz - 1, NULL, 0, Bc, bc_len) == -3,
        "nzval_len one element short returns -3");
  CHECK(all_sentinel(Bc, bc_len), "a short nzval_len writes nothing");
  CHECK(smc_compress(result, nzval, 0, NULL, 0, Bc, bc_len) == -3, "nzval_len 0 returns -3");
  CHECK(all_sentinel(Bc, bc_len), "an nzval_len of 0 writes nothing");

  CHECK(smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, bc_len - 1) == -3,
        "Bc_len one element short returns -3");
  CHECK(all_sentinel(Bc, bc_len), "a short Bc_len writes nothing");
  CHECK(smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, 0) == -3, "Bc_len 0 returns -3");
  CHECK(all_sentinel(Bc, bc_len), "a Bc_len of 0 writes nothing");

  CHECK(smc_compress(result, NULL, (size_t) nnz, NULL, 0, Bc, bc_len) == -3,
        "NULL nzval returns -3");
  CHECK(smc_compress(result, nzval, (size_t) nnz, NULL, 0, NULL, bc_len) == -3,
        "NULL Bc returns -3");
  CHECK(all_sentinel(Bc, bc_len), "no NULL-buffer rejection writes anything");

  /* A larger length is a generous promise, not an error: the comparison is
     unsigned and must not wrap. */
  CHECK(smc_compress(result, nzval, SIZE_MAX, NULL, 0, Bc, SIZE_MAX) == 0,
        "SIZE_MAX lengths do not wrap into 'too small'");

  /* Bc now holds the real compressed matrix. */
  fill_sentinel(A, a_len);
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, a_len - 1) == -3,
        "A_len one element short returns -3");
  CHECK(all_sentinel(A, a_len), "a short A_len writes nothing");
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, 0) == -3, "A_len 0 returns -3");
  CHECK(all_sentinel(A, a_len), "an A_len of 0 writes nothing");

  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len - 1, A, a_len) == -3,
        "Bc_len one element short returns -3 in decompress");
  CHECK(all_sentinel(A, a_len), "a short Bc_len leaves A_out alone");
  CHECK(smc_decompress(result, NULL, 0, NULL, bc_len, A, a_len) == -3,
        "NULL Bc returns -3 in decompress");
  CHECK(all_sentinel(A, a_len), "a NULL Bc leaves A_out alone");
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, NULL, a_len) == -3,
        "NULL A_out returns -3");

  CHECK(smc_decompress(result, NULL, 0, Bc, SIZE_MAX, A, SIZE_MAX) == 0,
        "SIZE_MAX lengths are accepted by decompress too");

  free(Bc); free(A);
  smc_result_free(result);
}

/* Bidirectional partition: Br is required, so all four lengths are in play. */
static void test_bidirectional_buffer_lengths(void)
{
  SmcColoringOptions o = smc_default_options();
  void *result = NULL;
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1;
  int nnz = -1, m = -1, n = -1;
  size_t br_len, bc_len, a_len;
  double *Br, *Bc, *A;

  printf("buffer lengths (bidirectional partition) ...\n");
  o.partition = SMC_BIDIRECTIONAL;
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &result) == 0, "bidirectional coloring");
  CHECK(smc_nnz(result, &nnz) == 0 && nnz == NNZ, "nnz");
  CHECK(smc_size(result, &m, &n) == 0 && m == M && n == N, "size");
  CHECK(smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols) == 0,
        "compressed size");
  CHECK(Br_rows > 0 && Br_cols == N, "a bidirectional partition does have a Br");

  br_len = (size_t) Br_rows * (size_t) Br_cols;
  bc_len = (size_t) Bc_rows * (size_t) Bc_cols;
  a_len = (size_t) m * (size_t) n;
  Br = (double *) malloc(sizeof(double) * br_len);
  Bc = (double *) malloc(sizeof(double) * bc_len);
  A = (double *) malloc(sizeof(double) * a_len);
  CHECK(Br != NULL && Bc != NULL && A != NULL, "allocation");
  if (Br == NULL || Bc == NULL || A == NULL) {
    free(Br); free(Bc); free(A); smc_result_free(result); return;
  }

  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, br_len, Bc, bc_len) == 0,
        "exact-size compress succeeds");
  CHECK(smc_decompress(result, Br, br_len, Bc, bc_len, A, a_len) == 0,
        "exact-size decompress succeeds");

  fill_sentinel(Br, br_len);
  fill_sentinel(Bc, bc_len);
  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, br_len - 1, Bc, bc_len) == -3,
        "Br_len one element short returns -3");
  CHECK(all_sentinel(Br, br_len) && all_sentinel(Bc, bc_len),
        "a short Br_len writes nothing, in either buffer");
  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, 0, Bc, bc_len) == -3,
        "Br_len 0 returns -3 for a bidirectional result");
  CHECK(all_sentinel(Br, br_len) && all_sentinel(Bc, bc_len), "a Br_len of 0 writes nothing");

  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, br_len, Bc, bc_len - 1) == -3,
        "Bc_len one element short returns -3");
  CHECK(smc_compress(result, nzval, (size_t) nnz - 1, Br, br_len, Bc, bc_len) == -3,
        "nzval_len one element short returns -3");
  CHECK(all_sentinel(Br, br_len) && all_sentinel(Bc, bc_len),
        "neither short length writes anything");

  /* Both compressed matrices are required. */
  CHECK(smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, bc_len) == -3,
        "a bidirectional result rejects a NULL Br");
  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, br_len, NULL, 0) == -3,
        "a bidirectional result rejects a NULL Bc");
  CHECK(all_sentinel(Br, br_len) && all_sentinel(Bc, bc_len), "no NULL rejection writes");

  CHECK(smc_compress(result, nzval, (size_t) nnz, Br, br_len, Bc, bc_len) == 0,
        "compress again, for the decompress checks");

  fill_sentinel(A, a_len);
  CHECK(smc_decompress(result, Br, br_len - 1, Bc, bc_len, A, a_len) == -3,
        "Br_len one element short returns -3 in decompress");
  CHECK(all_sentinel(A, a_len), "a short Br_len leaves A_out alone");
  CHECK(smc_decompress(result, Br, br_len, Bc, bc_len - 1, A, a_len) == -3,
        "Bc_len one element short returns -3 in decompress");
  CHECK(all_sentinel(A, a_len), "a short Bc_len leaves A_out alone");
  CHECK(smc_decompress(result, Br, br_len, Bc, bc_len, A, a_len - 1) == -3,
        "A_len one element short returns -3 in decompress");
  CHECK(all_sentinel(A, a_len), "a short A_len leaves A_out alone");
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, a_len) == -3,
        "a bidirectional result rejects a NULL Br in decompress");
  CHECK(all_sentinel(A, a_len), "a NULL Br leaves A_out alone");

  CHECK(smc_decompress(result, Br, br_len, Bc, bc_len, A, a_len) == 0,
        "the exact sizes still work after every rejection");

  free(Br); free(Bc); free(A);
  smc_result_free(result);
}

/* index_base shifts the pattern indices and the group members, never the
   colors: same matrix, same coloring. */
static void test_index_base(void)
{
  SmcColoringOptions o0 = smc_default_options();
  SmcColoringOptions o1 = smc_default_options();
  void *r0 = NULL, *r1 = NULL;
  int c0[N], c1[N], nc0 = 0, nc1 = 0, g0 = 0, g1 = 0, j, same = 1, shifted = 1;

  printf("index_base 0 vs 1 ...\n");
  o1.index_base = 1;

  CHECK(smc_coloring(M, N, colptr0, rowval0, &o0, &r0) == 0, "0-based coloring");
  CHECK(smc_coloring(M, N, colptr1, rowval1, &o1, &r1) == 0, "1-based coloring");

  smc_ncolors(r0, &nc0);
  smc_ncolors(r1, &nc1);
  CHECK(nc0 == nc1 && nc0 > 0, "same number of colors");

  smc_column_colors(r0, c0, N);
  smc_column_colors(r1, c1, N);
  for (j = 0; j < N; j++) if (c0[j] != c1[j]) same = 0;
  CHECK(same, "identical column colors (colors are labels, not indices)");

  smc_ncolumn_groups(r0, &g0);
  smc_ncolumn_groups(r1, &g1);
  CHECK(g0 == g1, "same number of groups");
  for (j = 1; j <= g0; j++) {
    int s0 = 0, s1 = 0, k, m0[N], m1[N];
    smc_column_group_size(r0, j, &s0);
    smc_column_group_size(r1, j, &s1);
    if (s0 != s1) { shifted = 0; break; }
    smc_column_group(r0, j, m0, s0);
    smc_column_group(r1, j, m1, s1);
    for (k = 0; k < s0; k++) if (m1[k] != m0[k] + 1) shifted = 0;
  }
  CHECK(shifted, "group members are shifted by exactly the index base");

  smc_result_free(r0);
  smc_result_free(r1);
}

static void test_float32(void)
{
  SmcColoringOptions o = smc_default_options();
  void *result = NULL;
  int Br_rows = -1, Br_cols = -1, Bc_rows = -1, Bc_cols = -1, i, j, k, exact = 1;
  int nnz = -1, m = -1, n = -1, intact = 1;
  size_t bc_len, a_len;
  float nzval32[NNZ], *Bc, A[M * N], expected[M * N];

  printf("Float32 compress / decompress ...\n");
  o.dtype = SMC_FLOAT32;
  CHECK(smc_coloring(M, N, colptr0, rowval0, &o, &result) == 0, "Float32 coloring");
  CHECK(smc_nnz(result, &nnz) == 0 && nnz == NNZ, "nnz");
  CHECK(smc_size(result, &m, &n) == 0 && m == M && n == N, "size");
  CHECK(smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols) == 0,
        "compressed size");
  CHECK(Br_rows == 0 && Br_cols == 0, "Br is unused for a column partition");
  CHECK(Bc_rows == M, "Bc has m rows");

  /* Lengths are element counts, so they match the Float64 numbers. */
  bc_len = (size_t) Bc_rows * (size_t) Bc_cols;
  a_len = (size_t) m * (size_t) n;

  for (k = 0; k < NNZ; k++) nzval32[k] = (float) nzval[k];
  Bc = (float *) malloc(sizeof(float) * bc_len);
  CHECK(Bc != NULL, "allocation");
  CHECK(smc_compress(result, nzval32, (size_t) nnz, NULL, 0, Bc, bc_len) == 0,
        "Float32 compress");
  CHECK(smc_compress(result, nzval32, (size_t) nnz - 1, NULL, 0, Bc, bc_len) == -3,
        "a short nzval_len is rejected on the Float32 path too");
  CHECK(smc_compress(result, nzval32, (size_t) nnz, NULL, 0, Bc, bc_len - 1) == -3,
        "a short Bc_len is rejected on the Float32 path too");

  for (k = 0; k < M * N; k++) A[k] = -987.0f;
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, a_len - 1) == -3,
        "a short A_len is rejected on the Float32 path too");
  for (k = 0; k < M * N; k++) if (A[k] != -987.0f) intact = 0;
  CHECK(intact, "a short A_len leaves the Float32 A_out untouched");
  CHECK(smc_decompress(result, NULL, 0, Bc, bc_len, A, a_len) == 0, "Float32 decompress");

  /* Reference dense matrix, column-major. */
  for (k = 0; k < M * N; k++) expected[k] = 0.0f;
  for (j = 0; j < N; j++)
    for (k = colptr0[j]; k < colptr0[j + 1]; k++)
      expected[j * M + rowval0[k]] = (float) nzval[k];

  for (j = 0; j < N; j++)
    for (i = 0; i < M; i++)
      if (A[j * M + i] != expected[j * M + i]) exact = 0;
  CHECK(exact, "Float32 compress -> decompress reproduces every entry exactly");

  free(Bc);
  smc_result_free(result);
}

int main(void)
{
  build_one_based();

  test_abi();
  test_default_options();
  test_version();
  test_null_options();
  test_unsupported_combinations();
  test_invalid_arguments();
  test_invalid_handle();
  test_sizing_queries();
  test_buffer_lengths();
  test_bidirectional_buffer_lengths();
  test_index_base();
  test_float32();

  printf("\n%d checks passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
