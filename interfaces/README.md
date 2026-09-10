# C and Fortran interfaces for SparseMatrixColorings.jl

Exposes [SparseMatrixColorings.jl](https://github.com/JuliaDiff/SparseMatrixColorings.jl) (SMC) as a
native shared library (`libsmc.so` / `libsmc.dylib` / `libsmc.dll`) callable from C, Fortran, and any
language with a C FFI.

Two bindings ship with the library: the C header [`include/smc.h`](include/smc.h) and the Fortran
include file [`include/smc.f90`](include/smc.f90) (see [Calling from Fortran](#calling-from-fortran)).

## Why

The library is built with [`juliac`](https://github.com/JuliaLang/JuliaC.jl) and `--trim=safe`, which
statically compiles the Julia code and bundles the Julia runtime next to it, giving a relocatable
directory that runs on a machine with no Julia:

```
interfaces/build/
├── lib/           (bin/ on Windows)
│   ├── libsmc.so  ← the library
│   └── julia/     ← embedded Julia runtime (no system Julia needed)
└── include/
    ├── smc.h      ← C header
    └── smc.f90    ← Fortran include file
```

## Prerequisites

| Tool | Version |
|------|---------|
| Julia | ≥ 1.12 |
| [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) | 0.3.8 (the version pinned by CI) |
| C compiler | gcc or clang (MSVC/MinGW on Windows) |
| Fortran compiler *(only to call it from Fortran)* | any Fortran 2003 compiler with `iso_c_binding` |

```bash
# Install JuliaC.jl once (installs juliac into ~/.julia/bin)
julia --startup-file=no -e 'import Pkg; Pkg.Registry.add("General"); Pkg.Apps.add(url="https://github.com/JuliaLang/JuliaC.jl", rev="v0.3.8")'
export PATH="$HOME/.julia/bin:$PATH"
```

## Build

All commands run from the root of the SparseMatrixColorings.jl repository.

```bash
julia --startup-file=no --project=. -e 'import Pkg; Pkg.instantiate()'
```

### Linux

```bash
mkdir -p interfaces/build/lib
juliac \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/build \
    --output-lib interfaces/build/lib/libsmc.so \
    interfaces/src/LibSMC.jl

# One run emits both interfaces/include/smc.h and interfaces/include/smc.f90
julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
mkdir -p interfaces/build/include
cp interfaces/include/smc.h interfaces/include/smc.f90 interfaces/build/include/
```

### macOS

```bash
mkdir -p interfaces/build/lib
juliac \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/build \
    --output-lib interfaces/build/lib/libsmc.dylib \
    interfaces/src/LibSMC.jl

julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
mkdir -p interfaces/build/include
cp interfaces/include/smc.h interfaces/include/smc.f90 interfaces/build/include/
```

### Windows (Git Bash / MSYS shell)

On Windows the bundle lands in `build/bin/` and the CLI is `juliac.bat`:

```bash
mkdir -p interfaces/build/bin
juliac.bat \
    --project . \
    --compile-ccallable \
    --trim=safe \
    --bundle interfaces/build \
    --output-lib interfaces/build/bin/libsmc.dll \
    interfaces/src/LibSMC.jl

julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
mkdir -p interfaces/build/include
cp interfaces/include/smc.h interfaces/include/smc.f90 interfaces/build/include/
```

> **Note.** The SuiteSparse libraries must be copied into the bundle by hand: `SparseArrays` pulls in
> `SuiteSparse_jll`, whose `__init__` `dlopen`s them, and `juliac` cannot trace a `dlopen`. Skip this
> and the first call aborts with
> `Core.InitError(mod=:SuiteSparse_jll, could not load library "libamd.so.3")`.
>
> ```bash
> JLIB="$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))')"
> cp -a "$JLIB"/lib{amd,btf,camd,ccolamd,cholmod,colamd,klu,ldl,rbio,spqr,suitesparseconfig,umfpack}.so* \
>       interfaces/build/lib/julia/
> ```
>
> (`.dylib` on macOS; on Windows the DLLs sit next to `julia.exe` and go into the bundle's `bin`.)
> CI does this in its "Bundle SuiteSparse libraries" step.

## Running the tests and examples

Set `LIBPATH` once; it is `interfaces/build/lib` everywhere except Windows, where it is
`interfaces/build/bin`.

```bash
LIBPATH="$(pwd)/interfaces/build/lib"  # Windows: .../interfaces/build/bin
```

### Julia tests

`interfaces/test/test_libsmc.jl` `include`s `interfaces/src/LibSMC.jl` as a plain Julia module, so it
needs no build. It must **not** `dlopen` the compiled library: that would start a second Julia
runtime inside the first and crash.

```bash
julia --startup-file=no --project=. interfaces/test/test_libsmc.jl
```

### C tests

```bash
gcc -O2 -o interfaces/build/test_api_c \
    interfaces/test/C/test_api.c \
    -I interfaces/include "$LIBPATH/libsmc.so" -lm

gcc -O2 -o interfaces/build/test_coloring_c \
    interfaces/test/C/test_coloring.c \
    -I interfaces/include "$LIBPATH/libsmc.so" -lm
```

### Fortran tests

`smc.f90` is an *include file*, not a module, so `-I interfaces/include` is all `gfortran` needs and
no `.mod` file is produced or consumed.

```bash
gfortran -O2 -o interfaces/build/test_smc_fortran \
    interfaces/test/Fortran/test_smc.f90 \
    -I interfaces/include "$LIBPATH/libsmc.so"
```

### Examples

```bash
gcc -O2 -o interfaces/build/basic_coloring \
    interfaces/examples/C/basic_coloring.c \
    -I interfaces/include "$LIBPATH/libsmc.so"

gcc -O2 -o interfaces/build/compress_decompress \
    interfaces/examples/C/compress_decompress.c \
    -I interfaces/include "$LIBPATH/libsmc.so" -lm

gcc -O2 -o interfaces/build/symmetric_coloring \
    interfaces/examples/C/symmetric_coloring.c \
    -I interfaces/include "$LIBPATH/libsmc.so"
```

The Fortran examples all build the same way:

```bash
for src in interfaces/examples/Fortran/*.f90; do
  name=$(basename "$src" .f90)
  gfortran -O2 -o "interfaces/build/${name}_fortran" \
      "$src" -I interfaces/include "$LIBPATH/libsmc.so"
done
```

On macOS replace `libsmc.so` with `libsmc.dylib`; on Windows with `libsmc.dll`. This holds for
`gfortran` too.

### Running

The loader must find `libsmc` *and* the bundled Julia runtime in `$LIBPATH/julia`. The dependency on
`julia/` is recorded in the library's own runpath, so putting `$LIBPATH` on the search path is
enough. Windows has no rpath — use `PATH`.

```bash
# Linux
export LD_LIBRARY_PATH="$LIBPATH:${LD_LIBRARY_PATH:-}"

# macOS
export DYLD_LIBRARY_PATH="$LIBPATH:${DYLD_LIBRARY_PATH:-}"

# Windows (Git Bash)
export PATH="$LIBPATH:$PATH"
```

```bash
# Linux
gcc -O2 -o interfaces/build/basic_coloring \
    interfaces/examples/C/basic_coloring.c \
    -I interfaces/build/include interfaces/build/lib/libsmc.so \
    -Wl,-rpath,'$ORIGIN/lib'

gfortran -O2 -o interfaces/build/basic_coloring_fortran \
    interfaces/examples/Fortran/basic_coloring.f90 \
    -I interfaces/build/include interfaces/build/lib/libsmc.so \
    -Wl,-rpath,'$ORIGIN/lib'

# macOS ($ORIGIN is spelled @loader_path)
clang -O2 -o interfaces/build/basic_coloring \
    interfaces/examples/C/basic_coloring.c \
    -I interfaces/build/include interfaces/build/lib/libsmc.dylib \
    -Wl,-rpath,@loader_path/lib
```

## Hello world (C)

```c
/* hello_smc.c — color a 4x4 tridiagonal matrix, no handle, no allocation. */
#include <stdio.h>
#include "smc.h"

int main(void)
{
  /* CSC, 0-based: column j is colptr[j] .. colptr[j+1]-1 */
  const int colptr[5]  = { 0, 2, 5, 8, 10 };
  const int rowval[10] = { 0, 1,  0, 1, 2,  1, 2, 3,  2, 3 };

  SmcColoringOptions opts = smc_default_options();  /* nonsymmetric / column / direct */

  int colors[4], ncolors;
  int ret = smc_fast_coloring(4, 4, colptr, rowval, &opts,
                              NULL,      /* row colors: only needed for SMC_BIDIRECTIONAL */
                              colors, &ncolors);
  if (ret != 0) { fprintf(stderr, "smc_fast_coloring failed (%d)\n", ret); return 1; }

  printf("ncolors = %d, colors =", ncolors);       /* ncolors = 3, colors = 1 2 3 1 */
  for (int j = 0; j < 4; j++) printf(" %d", colors[j]);
  printf("\n");
  return 0;
}
```

```bash
gcc -O2 -o hello_smc hello_smc.c -I interfaces/include "$LIBPATH/libsmc.so"
LD_LIBRARY_PATH="$LIBPATH" ./hello_smc
```

`smc_fast_coloring` is the stateless shortcut. When you also need groups, compression or
decompression, use `smc_coloring`, which returns an opaque handle to be released with
`smc_result_free`; see `examples/C/compress_decompress.c`.

## Calling from Fortran

`include/smc.f90` is the Fortran binding: the `SmcColoringOptions` derived type, the enumerators,
`SMC_VERSION_*` as `integer(c_int), parameter`, and an `interface` block declaring all 19 entry
points. It is **generated from the same table as `smc.h`**, so the two can never disagree.

It is an *include file* and not a module because a module would force us to ship a `.mod` file, and
`.mod` files are compiler- and version-specific (a gfortran 13 `.mod` is unreadable by ifx, or by
gfortran 9). Put `include 'smc.f90'` at the top of the specification part, after `implicit none` —
the type must be defined before you declare a variable of it:

```fortran
program my_prog
  use iso_c_binding
  implicit none
  include 'smc.f90'          ! <- here, after implicit none

  type(SmcColoringOptions), target :: opts
  ...
end program my_prog
```

### Hello world (Fortran)

```fortran
! hello_smc.f90 — color a 4x4 tridiagonal matrix, no handle, no allocation.
program hello_smc
  use iso_c_binding
  implicit none
  include 'smc.f90'

  ! CSC, 0-based: column j holds rowval(colptr(j)+1 .. colptr(j+1))
  integer(c_int), target :: colptr(5)  = [0, 2, 5, 8, 10]
  integer(c_int), target :: rowval(10) = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3]
  integer(c_int), target :: colors(4), ncolors
  type(SmcColoringOptions), target :: opts
  integer(c_int) :: ret

  opts = smc_default_options()          ! nonsymmetric / column / direct

  ret = smc_fast_coloring(4, 4, c_loc(colptr), c_loc(rowval), c_loc(opts), &
                          c_null_ptr,        &  ! row colors: only for SMC_BIDIRECTIONAL
                          c_loc(colors), c_loc(ncolors))
  if (ret /= 0) then
    print '(a,i0,a)', 'smc_fast_coloring failed (', ret, ')'
    stop 1
  end if

  print '(a,i0,a,4i2)', 'ncolors = ', ncolors, ', colors =', colors
end program hello_smc
```

```bash
gfortran -O2 -o hello_smc hello_smc.f90 -I interfaces/include "$LIBPATH/libsmc.so"
LD_LIBRARY_PATH="$LIBPATH" ./hello_smc          # ncolors = 3, colors = 1 2 3 1
```

`-I` points the compiler at the directory holding `smc.f90` (`interfaces/include`, or
`interfaces/build/include` in an unpacked bundle). Platform handling is exactly as for C:

```bash
gfortran -O2 -o interfaces/build/hello_smc hello_smc.f90 \
    -I interfaces/build/include interfaces/build/lib/libsmc.so \
    -Wl,-rpath,'$ORIGIN/lib'                    # macOS: -Wl,-rpath,@loader_path/lib
```

### The `c_loc` convention

Every C pointer is bound as `type(c_ptr), value`, so the caller passes `c_loc(x)` — and **`x` must
carry the `target` attribute**. Where C accepts `NULL`, pass `c_null_ptr`. The single exception is
the `void **result_out` of `smc_coloring`, an *out*-parameter, declared `type(c_ptr), intent(out)`
and passed as a bare `type(c_ptr)` variable:

```fortran
type(c_ptr) :: res
ret = smc_coloring(m, n, c_loc(colptr), c_loc(rowval), c_loc(opts), res)  ! no c_loc on res
...
ret = smc_result_free(res)
```

The full mapping:

| C | Fortran | at the call site |
|---|---|---|
| `int` | `integer(c_int), value` | a literal such as `4`, or an `integer(c_int)` variable |
| `size_t` | `integer(c_size_t), value` | `int(nnz, c_size_t)`, `0_c_size_t` |
| `int *`, `const int *` | `type(c_ptr), value` | `c_loc(colptr)` |
| `void *`, `const void *` | `type(c_ptr), value` | `c_loc(Bc)` or `c_null_ptr` |
| `void **` (out) | `type(c_ptr), intent(out)` | a plain `type(c_ptr)` variable |
| `const SmcColoringOptions *` | `type(c_ptr), value` | `c_loc(opts)` or `c_null_ptr` |
| returns `int` | `function ... result(ret)`, `integer(c_int) :: ret` | `ret = smc_...(...)` |
| returns `void` | `subroutine` | `call smc_version(...)` |
| returns `SmcColoringOptions` | `function ... result(opts)`, by value | `opts = smc_default_options()` |

Integer literals pass as `integer(c_int)` because the default integer kind is `c_int` — do not build
your program with `-fdefault-integer-8`, or declare the argument explicitly.

Scalar out-parameters are `int *` like any other pointer, so they too go through `c_loc` of a
`target` scalar — except `smc_version`, whose three outputs are plain `intent(out)` scalars:

```fortran
integer(c_int), target :: nc, m, n
integer(c_int)         :: major, minor, patch   ! no `target` needed

call smc_version(major, minor, patch)
ret = smc_ncolors(res, c_loc(nc))
ret = smc_size(res, c_loc(m), c_loc(n))
```

### Index base

`opts%index_base` defaults to **0**, the C convention. Fortran callers will usually want:

```fortran
opts = smc_default_options()
opts%index_base = 1
```

One setting covers both directions: it makes `colptr` and `rowval` 1-based *and* makes the group
members returned by `smc_column_group` / `smc_row_group` 1-based. Your arrays are never mutated.
Color *labels* are not indices and are never shifted: they are always `1..ncolors`, with `0` reserved
for "needs no evaluation" (only possible when `opts%postprocessing` is 1).

### Buffers

Dense buffers are column-major, already Fortran's layout, so a rank-2 array is handed over directly
with `c_loc` and no transpose. Buffer lengths are element counts, never bytes; the `size_t` ones need
an explicit kind, `int(Bc_rows * Bc_cols, c_size_t)`. For a non-bidirectional result `Br` is unused —
pass `c_null_ptr` with `0_c_size_t`:

```fortran
ret = smc_compress(res, c_loc(nzval), int(nnz, c_size_t), &
                   c_null_ptr, 0_c_size_t,                &
                   c_loc(Bc), int(Bc_rows * Bc_cols, c_size_t))
```

Worked programs: `examples/Fortran/basic_coloring.f90` and
`examples/Fortran/compress_decompress.f90`.

## Buffers and sizing

Every buffer is caller-allocated, and **every buffer is passed with its length**. Lengths are
*element counts*, never bytes — elements of the type selected by `opts.dtype` at coloring time
(`double` for `SMC_FLOAT64`, `float` for `SMC_FLOAT32`). A buffer shorter than its required minimum
is rejected with `-3` before a single element is read or written, which is what makes it safe to
reuse a handle with a different set of buffers. A length of `0` is legitimate.

Every one of those minimums can be queried from the handle alone:

| buffer | passed to | minimum length | query |
|---|---|---|---|
| `colors` | `smc_column_colors` / `smc_row_colors` | `n` / `m` | `smc_size` |
| `members` | `smc_column_group` / `smc_row_group` | the group size | `smc_column_group_size` / `smc_row_group_size` |
| `nzval` | `smc_compress` | `nnz` | `smc_nnz` |
| `Bc` | `smc_compress` / `smc_decompress` | `Bc_rows * Bc_cols` | `smc_compressed_size` |
| `Br` | `smc_compress` / `smc_decompress` | `Br_rows * Br_cols`, `0` unless bidirectional | `smc_compressed_size` |
| `A_out` | `smc_decompress` | `m * n` | `smc_size` |

The two queries that describe the pattern the result was built from:

```c
/* Number of stored entries; exactly the length nzval must have. */
int smc_nnz(void *result, int *nnz_out);

/* Dimensions of the pattern; A_out must hold m*n elements. */
int smc_size(void *result, int *m_out, int *n_out);
```

The compression entry points take one length per buffer, immediately after it:

```c
int smc_compress(void *result,
                 const void *nzval, size_t nzval_len,
                 void *Br, size_t Br_len,
                 void *Bc, size_t Bc_len);

int smc_decompress(void *result,
                   const void *Br, size_t Br_len,
                   const void *Bc, size_t Bc_len,
                   void *A_out, size_t A_len);
```

A result is bidirectional exactly when `Br_cols > 0`. Otherwise `Br` is unused: pass `NULL` with
`Br_len` 0. A bidirectional result needs both `Br` and `Bc`, and a `NULL` for either is `-3`. In
outline:

```c
int nnz, m, n, Br_rows, Br_cols, Bc_rows, Bc_cols;
smc_nnz(result, &nnz);
smc_size(result, &m, &n);
smc_compressed_size(result, &Br_rows, &Br_cols, &Bc_rows, &Bc_cols);

size_t Bc_len = (size_t)Bc_rows * (size_t)Bc_cols;
size_t A_len  = (size_t)m * (size_t)n;          /* m*n overflows a 32-bit int */
double *Bc = malloc(Bc_len * sizeof(double));
double *A  = malloc(A_len  * sizeof(double));

smc_compress(result, nzval, (size_t)nnz, NULL, 0, Bc, Bc_len);
smc_decompress(result, NULL, 0, Bc, Bc_len, A, A_len);
```

The color and group buffer lengths stay `int`, being bounded by `m` or `n`. The numerical buffers use
`size_t` because `m*n` overflows a 32-bit `int` at perfectly ordinary dimensions (`m = n = 50000`
already gives 2.5e9). `examples/C/compress_decompress.c` is the worked version of the sketch above.

## Return codes

Every entry point returns an `int`:

| code | meaning |
|-----:|---------|
| `0`  | success |
| `-1` | internal error (a Julia exception was caught and logged) |
| `-2` | unsupported combination of `(structure, partition, decompression, dtype)` |
| `-3` | invalid argument (NULL pointer, bad dimension, buffer too small, bad enum, bad `index_base`) |
| `-4` | invalid or already-freed handle |

Use-after-free and double-free of a handle return `-4` instead of crashing.

## Regenerating the headers

`include/smc.h` and `include/smc.f90` are both generated by a single run of the same script from a
single signature table. Both are committed and checked by CI (`git diff --exit-code` after a
regeneration, plus a check that the `SMC_VERSION_*` values match `Project.toml`). Never edit either
by hand — edit `interfaces/scripts/coloring_table.jl` or the signature table in
`interfaces/src/LibSMC.jl`, then:

```bash
julia --startup-file=no --project=. interfaces/scripts/generate_header.jl
git diff interfaces/include/smc.h interfaces/include/smc.f90
```

## Limitations

- **Only 6 combinations of `(structure, partition, decompression)` are supported**; anything else
  returns `-2`.

  | structure | partition | decompression | dtype |
  |---|---|---|---|
  | `SMC_NONSYMMETRIC` | `SMC_COLUMN` | `SMC_DIRECT` | both |
  | `SMC_NONSYMMETRIC` | `SMC_ROW` | `SMC_DIRECT` | both |
  | `SMC_NONSYMMETRIC` | `SMC_BIDIRECTIONAL` | `SMC_DIRECT` | `SMC_FLOAT64`, `SMC_FLOAT32` |
  | `SMC_NONSYMMETRIC` | `SMC_BIDIRECTIONAL` | `SMC_SUBSTITUTION` | `SMC_FLOAT64`, `SMC_FLOAT32` |
  | `SMC_SYMMETRIC` | `SMC_COLUMN` | `SMC_DIRECT` | both |
  | `SMC_SYMMETRIC` | `SMC_COLUMN` | `SMC_SUBSTITUTION` | `SMC_FLOAT64`, `SMC_FLOAT32` |

  In particular `(SMC_NONSYMMETRIC, SMC_COLUMN or SMC_ROW, SMC_SUBSTITUTION)` is not a valid SMC
  problem, and symmetric problems only exist with the column partition.
- **`RandomOrder` is not exposed.** It carries an `AbstractRNG`, which is untested under `--trim`.
  The five deterministic orders of `SmcOrder` are available.
- **Indices are 32-bit `int`.** They are widened to `Int64` internally, but matrices with more than
  2^31 stored entries are out of scope. Use `opts.index_base` to choose 0- or 1-based `colptr` /
  `rowval` (default 0); the caller's arrays are never mutated.
- **Values are `double` or `float` only** (`SmcDataType`); no complex, no `BigFloat`. Coloring itself
  is structure-only, so `dtype` matters only for compression / decompression.
- **Dense buffers are column-major** (`B[i,j] == B[i + j*ld]`), matching Julia and Fortran.
- **CPU only.** SMC's GPU support lives in Julia package extensions, which are not compiled into the
  library.
- **The Fortran binding is an include file, not a module**, so every pointer argument is a
  `type(c_ptr)` you fill with `c_loc(...)` rather than an assumed-shape array; see
  [Calling from Fortran](#calling-from-fortran).
- **The bundle is large (~275 MB)**: `libsmc` itself is only about 5 MB, but the bundle embeds the
  Julia runtime, OpenBLAS, and the SuiteSparse libraries that `SparseArrays` loads at startup.
