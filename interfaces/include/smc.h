#ifndef SMC_H
#define SMC_H

#include <stddef.h> /* size_t */

/* Version */
#define SMC_VERSION_MAJOR 0
#define SMC_VERSION_MINOR 4
#define SMC_VERSION_PATCH 27

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * libsmc - C interface to SparseMatrixColorings.jl
 *
 * Typical use:
 *
 *   SmcColoringOptions opts = smc_default_options();
 *   opts.structure = SMC_NONSYMMETRIC;
 *   opts.partition = SMC_COLUMN;
 *   opts.order     = SMC_LARGEST_FIRST;
 *
 *   void *result;
 *   if (smc_coloring(m, n, colptr, rowval, &opts, &result) != 0) { ... }
 *
 *   int nc;
 *   smc_ncolors(result, &nc);
 *   int *colors = malloc(n * sizeof(int));
 *   smc_column_colors(result, colors, n);
 *
 *   int Br_rows, Br_cols, Bc_rows, Bc_cols;
 *   smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols);
 *   size_t Bc_len = (size_t) Bc_rows * (size_t) Bc_cols;
 *   double *Bc = malloc(Bc_len * sizeof(double));
 *
 *   int nnz;
 *   smc_nnz(result, &nnz);
 *   smc_compress(result, nzval, (size_t) nnz, NULL, 0, Bc, Bc_len);
 *
 *   smc_result_free(result);
 *
 * The sparsity pattern is passed in compressed sparse column form: colptr
 * has n+1 entries, rowval has colptr[n] - colptr[0] entries, and both use
 * the index base selected by opts.index_base (0 by default).  The caller's
 * arrays are copied, never modified.  Coloring is structure-only: the
 * numerical values are needed only by smc_compress.
 *
 * Indices crossing the interface are 32-bit int; matrices with more than
 * 2^31 nonzeros are out of scope.  Dense matrices are column-major, with
 * double or float elements according to opts.dtype.
 *
 * Every output buffer is caller-allocated and every buffer argument is
 * immediately followed by its length, counted in elements, never in bytes.
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * Enumerators
 * ------------------------------------------------------------------------- */

/*
 * Element type of the numerical buffers passed to smc_compress and
 * smc_decompress (double or float).  The sparsity pattern is always int.
 */
typedef enum {
  SMC_FLOAT64 = 0,
  SMC_FLOAT32 = 1
} SmcDataType;

/*
 * Structure of the matrix.  SMC_SYMMETRIC states that the sparsity pattern
 * is symmetric and selects the symmetric coloring problems.
 */
typedef enum {
  SMC_NONSYMMETRIC = 0,
  SMC_SYMMETRIC    = 1
} SmcStructure;

/*
 * Which dimension is colored.  SMC_BIDIRECTIONAL colors rows and columns at
 * the same time and produces two compressed matrices.
 */
typedef enum {
  SMC_COLUMN        = 0,
  SMC_ROW           = 1,
  SMC_BIDIRECTIONAL = 2
} SmcPartition;

/*
 * How the nonzeros are recovered from the compressed matrix.
 * SMC_SUBSTITUTION needs fewer colors but is only available for the
 * symmetric-column and bidirectional problems.
 */
typedef enum {
  SMC_DIRECT       = 0,
  SMC_SUBSTITUTION = 1
} SmcDecompression;

/*
 * Vertex order used by the greedy coloring algorithm.
 * RandomOrder is deliberately not exposed by this interface.
 */
typedef enum {
  SMC_NATURAL               = 0,
  SMC_LARGEST_FIRST         = 1,
  SMC_SMALLEST_LAST         = 2,
  SMC_INCIDENCE_DEGREE      = 3,
  SMC_DYNAMIC_LARGEST_FIRST = 4
} SmcOrder;

/* -------------------------------------------------------------------------
 * Coloring options
 *
 * Passed to smc_coloring and smc_fast_coloring, and remembered by the
 * result handle.  Initialise with smc_default_options() before overriding
 * individual fields; a NULL options pointer means the defaults.
 *
 * Supported (structure, partition, decompression) combinations; anything
 * else is rejected with -2:
 *   SMC_NONSYMMETRIC SMC_COLUMN        SMC_DIRECT
 *   SMC_NONSYMMETRIC SMC_ROW           SMC_DIRECT
 *   SMC_SYMMETRIC    SMC_COLUMN        SMC_DIRECT
 *   SMC_SYMMETRIC    SMC_COLUMN        SMC_SUBSTITUTION
 *   SMC_NONSYMMETRIC SMC_BIDIRECTIONAL SMC_DIRECT
 *   SMC_NONSYMMETRIC SMC_BIDIRECTIONAL SMC_SUBSTITUTION
 * ------------------------------------------------------------------------- */

typedef struct {
  int structure;          /* SmcStructure     - default SMC_NONSYMMETRIC                */
  int partition;          /* SmcPartition     - default SMC_COLUMN                      */
  int decompression;      /* SmcDecompression - default SMC_DIRECT                      */
  int order;              /* SmcOrder         - default SMC_NATURAL                     */
  int postprocessing;     /* 0/1 - give the neutral color 0 to the entries that need no */
                          /*       evaluation, where possible (default 0)               */
  int symmetric_pattern;  /* 0/1 - assert that the sparsity pattern is symmetric,       */
                          /*       skipping the symmetrization step (default 0)         */
  int index_base;         /* 0 or 1 - index base of colptr, rowval and of the group     */
                          /*          members returned by the queries (default 0)       */
  int dtype;              /* SmcDataType - element type used by smc_compress and        */
                          /*               smc_decompress (default SMC_FLOAT64)         */
} SmcColoringOptions;

/* -------------------------------------------------------------------------
 * Return codes
 *
 * Every function returning int returns one of:
 *
 *    0  success
 *   -1  internal error (a Julia exception was caught and logged)
 *   -2  unsupported combination of (structure, partition, decompression,
 *       dtype)
 *   -3  invalid argument (NULL pointer, bad dimension, buffer too small,
 *       bad enum value, bad index_base)
 *   -4  invalid or already-freed handle
 * ------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
 * API functions
 * ------------------------------------------------------------------------- */

/*
 * Return an SmcColoringOptions filled with the defaults: nonsymmetric
 * structure, column partition, direct decompression, natural order, no
 * postprocessing, no symmetric-pattern assertion, 0-based indices and
 * SMC_FLOAT64.  Always initialise an options struct with this call before
 * overriding individual fields.
 */
SmcColoringOptions smc_default_options(void);

/*
 * Write the SparseMatrixColorings.jl version of this library into
 * *major, *minor, *patch (the same values as the SMC_VERSION_* macros).
 */
void smc_version(int* major, int* minor, int* patch);

/* -------------------------------------------------------------------------
 * Coloring
 *
 * The pattern is always given in CSC form, in opts->index_base.  The
 * caller's arrays are copied and never modified.
 * ------------------------------------------------------------------------- */

/*
 * Color the m-by-n sparsity pattern given in CSC form and return an opaque
 * result handle through *result_out.  Only the structure is needed here;
 * the numerical values are passed later to smc_compress.
 *   m, n       : number of rows and columns, both > 0
 *   colptr     : n+1 column pointers, in opts->index_base
 *   rowval     : row indices of the nonzeros, in opts->index_base,
 *                length colptr[n] - colptr[0]
 *   opts       : coloring options, or NULL for smc_default_options()
 *   result_out : receives the handle; release it with smc_result_free
 * Returns 0, -1 on an internal error, -2 if the combination of structure,
 * partition, decompression and dtype is unsupported, -3 on an invalid
 * argument.
 */
int smc_coloring(int m, int n, const int* colptr, const int* rowval, const SmcColoringOptions* opts, void** result_out);

/*
 * Color the pattern and write the colors directly, without allocating a
 * handle.  Convenient when only the colors are needed; the groups and the
 * compression helpers require smc_coloring instead.
 *   row_colors    : length-m buffer, may be NULL when the partition
 *                   produces no row coloring (SMC_COLUMN)
 *   column_colors : length-n buffer, may be NULL when the partition
 *                   produces no column coloring (SMC_ROW)
 *   ncolors_out   : receives the number of colors
 * SMC_BIDIRECTIONAL fills both buffers, so neither may be NULL.
 * Colors are labels in 1..ncolors; 0 marks an entry that needs no
 * evaluation and can only appear when opts->postprocessing is 1.  Color
 * labels are never shifted by opts->index_base.
 * Returns 0, -1 on an internal error, -2 on an unsupported combination,
 * -3 on an invalid argument.
 */
int smc_fast_coloring(int m, int n, const int* colptr, const int* rowval, const SmcColoringOptions* opts, int* row_colors, int* column_colors, int* ncolors_out);

/*
 * Release a handle returned by smc_coloring; it must not be used again.
 * Returns 0, or -4 if the handle is unknown (freeing twice is safe).
 */
int smc_result_free(void* result);

/* -------------------------------------------------------------------------
 * Queries
 *
 * All of them take a handle from smc_coloring.  Every buffer crossing
 * the interface carries its own length, that length is checked before a
 * single element is read or written, and every sizing question has a
 * query, so a caller can always ask before allocating:
 *
 *   buffer   length argument   how to obtain the required length
 *   colors   len               n (columns) or m (rows), from smc_size
 *   members  len               smc_column_group_size / smc_row_group_size
 *   nzval    nzval_len         smc_nnz
 *   Bc       Bc_len            Bc_rows*Bc_cols, smc_compressed_size
 *   Br       Br_len            Br_rows*Br_cols, smc_compressed_size
 *   A_out    A_len             m*n, from smc_size
 *
 * Lengths are element counts, never byte counts.  The numerical
 * buffers use size_t rather than int because A_len is m*n, which
 * overflows a 32-bit int.  Too small a buffer is rejected with -3.
 * ------------------------------------------------------------------------- */

/*
 * Write the total number of colors of the result into *ncolors_out.
 * Returns 0, -3 on an invalid argument, -4 on an invalid handle.
 */
int smc_ncolors(void* result, int* ncolors_out);

/*
 * Copy the color of every column into `colors`; `len` must be at least n.
 * Colors are labels in 1..ncolors, 0 meaning "no evaluation needed".
 * Returns 0, -2 if the partition has no column coloring, -3 on an invalid
 * argument (including len < n), -4 on an invalid handle.
 */
int smc_column_colors(void* result, int* colors, int len);

/*
 * Copy the color of every row into `colors`; `len` must be at least m.
 * Colors are labels in 1..ncolors, 0 meaning "no evaluation needed".
 * Returns 0, -2 if the partition has no row coloring, -3 on an invalid
 * argument (including len < m), -4 on an invalid handle.
 */
int smc_row_colors(void* result, int* colors, int len);

/*
 * Write the number of column groups into *ngroups_out.  Groups are the
 * color classes: group g holds every column colored g.
 * Returns 0, -2 if the partition has no column coloring, -3 on an invalid
 * argument, -4 on an invalid handle.
 */
int smc_ncolumn_groups(void* result, int* ngroups_out);

/*
 * Write the number of row groups into *ngroups_out.
 * Returns 0, -2 if the partition has no row coloring, -3 on an invalid
 * argument, -4 on an invalid handle.
 */
int smc_nrow_groups(void* result, int* ngroups_out);

/*
 * Write the number of columns in column group `group` into *size_out.
 * `group` is 1-based and runs over 1..smc_ncolumn_groups, independently of
 * opts->index_base.  Query the size first, then fetch the members.
 * Returns 0, -2 if the partition has no column coloring, -3 on an invalid
 * argument (including an out-of-range group), -4 on an invalid handle.
 */
int smc_column_group_size(void* result, int group, int* size_out);

/*
 * Copy the column indices of column group `group` into `members`; `len`
 * must be at least smc_column_group_size(result, group).  The indices are
 * written in opts->index_base.
 * Returns 0, -2 if the partition has no column coloring, -3 on an invalid
 * argument (including len too small), -4 on an invalid handle.
 */
int smc_column_group(void* result, int group, int* members, int len);

/*
 * Write the number of rows in row group `group` into *size_out.  `group`
 * is 1-based and runs over 1..smc_nrow_groups.
 * Returns 0, -2 if the partition has no row coloring, -3 on an invalid
 * argument (including an out-of-range group), -4 on an invalid handle.
 */
int smc_row_group_size(void* result, int group, int* size_out);

/*
 * Copy the row indices of row group `group` into `members`; `len` must be
 * at least smc_row_group_size(result, group).  The indices are written in
 * opts->index_base.
 * Returns 0, -2 if the partition has no row coloring, -3 on an invalid
 * argument (including len too small), -4 on an invalid handle.
 */
int smc_row_group(void* result, int group, int* members, int len);

/*
 * Write the number of stored entries of the sparsity pattern this result
 * was built from into *nnz_out.  That is exactly the number of elements
 * `nzval` must have in smc_compress, i.e. the required nzval_len.
 * Returns 0, -3 on an invalid argument, -4 on an invalid handle.
 */
int smc_nnz(void* result, int* nnz_out);

/*
 * Write the dimensions of the matrix this result was built from into
 * *m_out and *n_out.  They are the lengths expected by smc_row_colors (m)
 * and smc_column_colors (n), and A_out in smc_decompress must hold m*n
 * elements, i.e. A_len must be at least m*n.
 * Both out pointers must be non-NULL.
 * Returns 0, -3 on an invalid argument, -4 on an invalid handle.
 */
int smc_size(void* result, int* m_out, int* n_out);

/* -------------------------------------------------------------------------
 * Compression / decompression
 *
 * Dense matrices are column-major (Fortran / Julia order) and hold
 * double or float elements according to opts->dtype.
 * ------------------------------------------------------------------------- */

/*
 * Report the dimensions of the compressed matrices, so the caller can size
 * the buffers of smc_compress and smc_decompress.
 *   Bc : m-by-ncolors for a column partition, ncolors-by-n for a row
 *        partition, m-by-ncolumn_groups for a bidirectional one
 *   Br : nrow_groups-by-n, and used only by SMC_BIDIRECTIONAL; for the
 *        other partitions *Br_rows and *Br_cols are set to 0
 * The Bc_len and Br_len arguments of smc_compress and smc_decompress must
 * be at least Bc_rows*Bc_cols and Br_rows*Br_cols respectively.
 * All four out pointers must be non-NULL.
 * Returns 0, -3 on an invalid argument, -4 on an invalid handle.
 */
int smc_compressed_size(void* result, int* Br_rows, int* Br_cols, int* Bc_rows, int* Bc_cols);

/*
 * Compress the matrix into the dense buffers Br and Bc.  Every buffer is
 * followed by its length, counted in elements of the type selected by
 * opts->dtype -- never in bytes.
 *   nzval, nzval_len : the CSC values, in the same order as the rowval
 *                      given to smc_coloring; double* or float* according
 *                      to opts->dtype.  nzval_len must be at least the
 *                      value reported by smc_nnz
 *   Br, Br_len       : row-compressed matrix, used only by
 *                      SMC_BIDIRECTIONAL; for the other partitions Br
 *                      may be NULL and Br_len 0.  A bidirectional result
 *                      requires it, with Br_len at least Br_rows*Br_cols
 *                      from smc_compressed_size
 *   Bc, Bc_len       : column-compressed matrix; Bc_len must be at least
 *                      Bc_rows*Bc_cols from smc_compressed_size
 * Both buffers are column-major with the dimensions reported by
 * smc_compressed_size: B[i,j] is B[j*rows + i].
 * Returns 0, -1 on an internal error, -3 on an invalid argument (a NULL
 * buffer the partition needs, or a buffer too small), -4 on an invalid
 * handle.
 */
int smc_compress(void* result, const void* nzval, size_t nzval_len, void* Br, size_t Br_len, void* Bc, size_t Bc_len);

/*
 * Recover the full m-by-n dense matrix from the compressed form.  Every
 * buffer is followed by its length, counted in elements of the type
 * selected by opts->dtype -- never in bytes.
 *   Br, Br_len   : the buffer filled by smc_compress, used only by
 *                  SMC_BIDIRECTIONAL; for the other partitions Br may be
 *                  NULL and Br_len 0.  A bidirectional result requires
 *                  it, with Br_len at least Br_rows*Br_cols from
 *                  smc_compressed_size
 *   Bc, Bc_len   : the buffer filled by smc_compress; Bc_len must be at
 *                  least Bc_rows*Bc_cols from smc_compressed_size
 *   A_out, A_len : m*n elements, column-major, of the type selected by
 *                  opts->dtype; A_out[i,j] is A_out[j*m + i].  A_len must
 *                  be at least m*n, with m and n from smc_size
 * Entries outside the sparsity pattern are set to zero.
 * Returns 0, -1 on an internal error, -3 on an invalid argument (a NULL
 * buffer the partition needs, or a buffer too small), -4 on an invalid
 * handle.
 */
int smc_decompress(void* result, const void* Br, size_t Br_len, const void* Bc, size_t Bc_len, void* A_out, size_t A_len);

#ifdef __cplusplus
}
#endif

#endif /* SMC_H */
