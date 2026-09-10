module LibSMC

using SparseArrays
using SparseMatrixColorings

# Matches the SMC_VERSION_* macros in smc.h; queryable at run time via smc_version.
const _SMC_VERSION = pkgversion(SparseMatrixColorings)

include(joinpath(@__DIR__, "..", "scripts", "coloring_table.jl"))
include("c_enums.jl")
include("c_stores.jl")

# Signatures exported to the generated C header, read by
# scripts/generate_header.jl.  Each entry: (c_name, return_type, [(arg, type)]).
const function_sigs = Tuple{String,String,Vector{Tuple{String,String}}}[]

macro export_sig(name, ret, args...)
    arg_pairs = [(string(a.args[1]), string(a.args[2])) for a in args]
    push!(function_sigs, (string(name), string(ret), arg_pairs))
    return esc(:(nothing))
end

# Only the twelve combo keys of `coloring_table.jl` have a store; every other
# well-formed combination is -2.  Spelled out rather than looked up in
# `SUPPORTED_KEYS` so that no container is touched on this path.
function _supported_key(key::UInt8)
    key == KEY_NS_COL_DIRECT_F64 && return true
    key == KEY_NS_COL_DIRECT_F32 && return true
    key == KEY_NS_ROW_DIRECT_F64 && return true
    key == KEY_NS_ROW_DIRECT_F32 && return true
    key == KEY_SYM_COL_DIRECT_F64 && return true
    key == KEY_SYM_COL_DIRECT_F32 && return true
    key == KEY_SYM_COL_SUBST_F64 && return true
    key == KEY_SYM_COL_SUBST_F32 && return true
    key == KEY_NS_BID_DIRECT_F64 && return true
    key == KEY_NS_BID_DIRECT_F32 && return true
    key == KEY_NS_BID_SUBST_F64 && return true
    key == KEY_NS_BID_SUBST_F32 && return true
    return false
end

# NULL selects the documented defaults.
function _load_options(opts_ptr::Ptr{Cvoid})
    opts_ptr == C_NULL && return SMC_DEFAULT_OPTIONS
    return unsafe_load(Ptr{SmcColoringOptionsC}(opts_ptr))
end

# Check the CSC arrays before anything reads them: `colptr` must start at the
# index base and be non-decreasing, and every row index must be inside 1..m.
# This is what makes the hand-rolled compression loops safe.
function _check_pattern(m::Int, n::Int, colptr::Ptr{Cint}, rowval::Ptr{Cint}, base::Int)
    start = Int(unsafe_load(colptr, 1))
    start == base || return Cint(-3)
    previous = start
    @inbounds for j in 2:(n + 1)
        p = Int(unsafe_load(colptr, j))
        p >= previous || return Cint(-3)
        previous = p
    end
    nz = previous - start
    # The API takes no `nnz` argument, so `nz` comes from the caller's
    # `colptr[n+1]`; without this bound a garbage value reads off the end of
    # `rowval`.  A duplicate-free CSC pattern holds at most `m * n` entries, and
    # `m * n` cannot overflow because both came from a `Cint`.
    nz <= m * n || return Cint(-3)
    @inbounds for k in 1:nz
        i = Int(unsafe_load(rowval, k)) - base + 1
        (1 <= i <= m) || return Cint(-3)
    end
    return Cint(0)
end

# Build the internal matrix directly from the caller's arrays, which are only
# read.  The index base is applied exactly once, here.  The values are
# irrelevant to a coloring, so they are all ones.
function _build_matrix(m::Int, n::Int, colptr::Ptr{Cint}, rowval::Ptr{Cint}, base::Int)
    shift = 1 - base
    cp = Vector{Int64}(undef, n + 1)
    @inbounds for j in 1:(n + 1)
        cp[j] = Int64(unsafe_load(colptr, j)) + shift
    end
    nz = Int(cp[n + 1] - cp[1])
    rv = Vector{Int64}(undef, nz)
    @inbounds for k in 1:nz
        rv[k] = Int64(unsafe_load(rowval, k)) + shift
    end
    nzv = ones(Float64, nz)
    return SparseMatrixCSC{Float64,Int64}(m, n, cp, rv, nzv)
end

# Order dispatch through a function barrier: a `Union{NaturalOrder,...}` hoisted
# into a variable defeats `--trim=safe`.  `@order_barrier order impl A rest...`
# expands to an if/elseif chain calling `impl(A, <concrete literal order>,
# rest...)`, so every branch gets its own specialization.

const _ORDER_EXPRS = (
    (:SMC_NATURAL, :(NaturalOrder())),
    (:SMC_LARGEST_FIRST, :(LargestFirst())),
    (:SMC_SMALLEST_LAST, :(DynamicDegreeBasedOrder{:back,:high2low,false}())),
    (:SMC_INCIDENCE_DEGREE, :(DynamicDegreeBasedOrder{:back,:low2high,false}())),
    (:SMC_DYNAMIC_LARGEST_FIRST, :(DynamicDegreeBasedOrder{:forward,:low2high,false}())),
)

macro order_barrier(order, impl, matrix, rest...)
    last_key, last_order = _ORDER_EXPRS[end]
    chain = Expr(:call, impl, matrix, last_order, rest...)
    for i in (length(_ORDER_EXPRS) - 1):-1:1
        keyname, order_expr = _ORDER_EXPRS[i]
        chain = Expr(
            :if,
            :($order == $keyname),
            Expr(:call, impl, matrix, order_expr, rest...),
            chain,
        )
    end
    return esc(chain)
end

# `postprocessing` is a literal in each branch; both build the same concrete
# `GreedyColoringAlgorithm` type, so the result is still statically known.
macro greedy(decompression, order, postprocessing)
    return esc(
        quote
            if $postprocessing
                GreedyColoringAlgorithm{$decompression}($order; postprocessing=true)
            else
                GreedyColoringAlgorithm{$decompression}($order; postprocessing=false)
            end
        end
    )
end

## nonsymmetric / column / direct

function _color_ns_col_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return coloring(
        A,
        ColoringProblem{:nonsymmetric,:column}(),
        algo;
        decompression_eltype=Float64,
        symmetric_pattern=sp,
    )
end

function _color_ns_col(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _color_ns_col_impl A pp sp
end

function _fast_ns_col_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return fast_coloring(
        A, ColoringProblem{:nonsymmetric,:column}(), algo; symmetric_pattern=sp
    )
end

function _fast_ns_col(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_ns_col_impl A pp sp
end

## nonsymmetric / row / direct

function _color_ns_row_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return coloring(
        A,
        ColoringProblem{:nonsymmetric,:row}(),
        algo;
        decompression_eltype=Float64,
        symmetric_pattern=sp,
    )
end

function _color_ns_row(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _color_ns_row_impl A pp sp
end

function _fast_ns_row_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return fast_coloring(
        A, ColoringProblem{:nonsymmetric,:row}(), algo; symmetric_pattern=sp
    )
end

function _fast_ns_row(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_ns_row_impl A pp sp
end

## symmetric / column / direct

function _color_sym_col_direct_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return coloring(
        A,
        ColoringProblem{:symmetric,:column}(),
        algo;
        decompression_eltype=Float64,
        symmetric_pattern=sp,
    )
end

function _color_sym_col_direct(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _color_sym_col_direct_impl A pp sp
end

function _fast_sym_col_direct_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return fast_coloring(
        A, ColoringProblem{:symmetric,:column}(), algo; symmetric_pattern=sp
    )
end

function _fast_sym_col_direct(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_sym_col_direct_impl A pp sp
end

## symmetric / column / substitution — `decompression_eltype` is part of the
## result type, so the element type travels as a type parameter.

function _color_sym_col_subst_impl(A, order, pp::Bool, sp::Bool, ::Type{R}) where {R}
    algo = @greedy :substitution order pp
    # `R` survives into the inferred result type only because `coloring` is marked
    # `Base.@constprop :aggressive` upstream (see src/interface.jl); otherwise the
    # keyword tuple widens `Type{R}` to `DataType`, `R` is erased and `--trim=safe`
    # rejects the call.  A typeassert on the result does not rescue it: it narrows
    # downstream of the call, while the verifier rejects the call itself.
    return coloring(
        A,
        ColoringProblem{:symmetric,:column}(),
        algo;
        decompression_eltype=R,
        symmetric_pattern=sp,
    )
end

function _color_sym_col_subst(A, order::Cint, pp::Bool, sp::Bool, ::Type{R}) where {R}
    @order_barrier order _color_sym_col_subst_impl A pp sp R
end

function _fast_sym_col_subst_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :substitution order pp
    return fast_coloring(
        A, ColoringProblem{:symmetric,:column}(), algo; symmetric_pattern=sp
    )
end

function _fast_sym_col_subst(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_sym_col_subst_impl A pp sp
end

## nonsymmetric / bidirectional / direct

function _color_ns_bid_direct_impl(A, order, pp::Bool, sp::Bool, ::Type{R}) where {R}
    algo = @greedy :direct order pp
    # `@constprop :aggressive` on `coloring` — see `_color_sym_col_subst_impl`.
    return coloring(
        A,
        ColoringProblem{:nonsymmetric,:bidirectional}(),
        algo;
        decompression_eltype=R,
        symmetric_pattern=sp,
    )
end

function _color_ns_bid_direct(A, order::Cint, pp::Bool, sp::Bool, ::Type{R}) where {R}
    @order_barrier order _color_ns_bid_direct_impl A pp sp R
end

function _fast_ns_bid_direct_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :direct order pp
    return fast_coloring(
        A, ColoringProblem{:nonsymmetric,:bidirectional}(), algo; symmetric_pattern=sp
    )
end

function _fast_ns_bid_direct(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_ns_bid_direct_impl A pp sp
end

## nonsymmetric / bidirectional / substitution

function _color_ns_bid_subst_impl(A, order, pp::Bool, sp::Bool, ::Type{R}) where {R}
    algo = @greedy :substitution order pp
    # `@constprop :aggressive` on `coloring` — see `_color_sym_col_subst_impl`.
    return coloring(
        A,
        ColoringProblem{:nonsymmetric,:bidirectional}(),
        algo;
        decompression_eltype=R,
        symmetric_pattern=sp,
    )
end

function _color_ns_bid_subst(A, order::Cint, pp::Bool, sp::Bool, ::Type{R}) where {R}
    @order_barrier order _color_ns_bid_subst_impl A pp sp R
end

function _fast_ns_bid_subst_impl(A, order, pp::Bool, sp::Bool)
    algo = @greedy :substitution order pp
    return fast_coloring(
        A, ColoringProblem{:nonsymmetric,:bidirectional}(), algo; symmetric_pattern=sp
    )
end

function _fast_ns_bid_subst(A, order::Cint, pp::Bool, sp::Bool)
    @order_barrier order _fast_ns_bid_subst_impl A pp sp
end

function _do_coloring(
    m::Cint,
    n::Cint,
    colptr::Ptr{Cint},
    rowval::Ptr{Cint},
    opts_ptr::Ptr{Cvoid},
    result_out::Ptr{Ptr{Cvoid}},
)
    result_out == C_NULL && return Cint(-3)
    (colptr == C_NULL || rowval == C_NULL) && return Cint(-3)
    (m > 0 && n > 0) || return Cint(-3)
    opts = _load_options(opts_ptr)
    _valid_options(opts) || return Cint(-3)
    key = combo_key(opts.structure, opts.partition, opts.decompression, opts.dtype)
    _supported_key(key) || return Cint(-2)

    base = Int(opts.index_base)
    rc = _check_pattern(Int(m), Int(n), colptr, rowval, base)
    rc == Cint(0) || return rc
    A = _build_matrix(Int(m), Int(n), colptr, rowval, base)

    order = opts.order
    pp = opts.postprocessing != Cint(0)
    sp = opts.symmetric_pattern != Cint(0)
    ib = opts.index_base

    # The `::` assertions are a safety net, not what makes this trim: they turn
    # any future drift of the store table into a loud TypeError
    # instead of a silent -1.  A green `test_libsmc.jl` does not imply a green
    # build -- the suite runs uncompiled and cannot see trim verifier errors, so
    # run the juliac build before trusting a change to these paths.
    if key == KEY_NS_COL_DIRECT_F64 || key == KEY_NS_COL_DIRECT_F32
        res = _color_ns_col(A, order, pp, sp)::SmcColumnResult
        return _register!(store_ns_col_direct, res, key, ib, result_out)
    elseif key == KEY_NS_ROW_DIRECT_F64 || key == KEY_NS_ROW_DIRECT_F32
        res = _color_ns_row(A, order, pp, sp)::SmcRowResult
        return _register!(store_ns_row_direct, res, key, ib, result_out)
    elseif key == KEY_SYM_COL_DIRECT_F64 || key == KEY_SYM_COL_DIRECT_F32
        res = _color_sym_col_direct(A, order, pp, sp)::SmcStarResult
        return _register!(store_sym_col_direct, res, key, ib, result_out)
    elseif key == KEY_SYM_COL_SUBST_F64
        res = _color_sym_col_subst(A, order, pp, sp, Float64)::SmcTreeResult{Float64}
        return _register!(store_sym_col_subst_f64, res, key, ib, result_out)
    elseif key == KEY_SYM_COL_SUBST_F32
        res = _color_sym_col_subst(A, order, pp, sp, Float32)::SmcTreeResult{Float32}
        return _register!(store_sym_col_subst_f32, res, key, ib, result_out)
    elseif key == KEY_NS_BID_DIRECT_F64
        res = _color_ns_bid_direct(A, order, pp, sp, Float64)::SmcBiDirectResult{Float64}
        return _register!(store_ns_bid_direct_f64, res, key, ib, result_out)
    elseif key == KEY_NS_BID_DIRECT_F32
        res = _color_ns_bid_direct(A, order, pp, sp, Float32)::SmcBiDirectResult{Float32}
        return _register!(store_ns_bid_direct_f32, res, key, ib, result_out)
    elseif key == KEY_NS_BID_SUBST_F64
        res = _color_ns_bid_subst(A, order, pp, sp, Float64)::SmcBiSubstResult{Float64}
        return _register!(store_ns_bid_subst_f64, res, key, ib, result_out)
    elseif key == KEY_NS_BID_SUBST_F32
        res = _color_ns_bid_subst(A, order, pp, sp, Float32)::SmcBiSubstResult{Float32}
        return _register!(store_ns_bid_subst_f32, res, key, ib, result_out)
    else
        return Cint(-2)
    end
end

# Write one color vector and the color count.  Color labels are never shifted
# by the index base: 0 is the neutral color, 1..ncolors are real colors.
function _emit_colors!(color, out::Ptr{Cint}, expected::Int, ncolors_out::Ptr{Cint})
    length(color) == expected || return Cint(-1)
    maxcolor = 0
    @inbounds for i in 1:expected
        ci = Int(color[i])
        unsafe_store!(out, Cint(ci), i)
        ci > maxcolor && (maxcolor = ci)
    end
    unsafe_store!(ncolors_out, Cint(maxcolor))
    return Cint(0)
end

# Bidirectional: both vectors, and `ncolors` is the sum of the two counts.
function _emit_colors2!(
    row_color,
    column_color,
    row_out::Ptr{Cint},
    column_out::Ptr{Cint},
    m::Int,
    n::Int,
    ncolors_out::Ptr{Cint},
)
    (length(row_color) == m && length(column_color) == n) || return Cint(-1)
    max_row = 0
    @inbounds for i in 1:m
        ci = Int(row_color[i])
        unsafe_store!(row_out, Cint(ci), i)
        ci > max_row && (max_row = ci)
    end
    max_column = 0
    @inbounds for j in 1:n
        cj = Int(column_color[j])
        unsafe_store!(column_out, Cint(cj), j)
        cj > max_column && (max_column = cj)
    end
    unsafe_store!(ncolors_out, Cint(max_row + max_column))
    return Cint(0)
end

function _do_fast_coloring(
    m::Cint,
    n::Cint,
    colptr::Ptr{Cint},
    rowval::Ptr{Cint},
    opts_ptr::Ptr{Cvoid},
    row_colors::Ptr{Cint},
    column_colors::Ptr{Cint},
    ncolors_out::Ptr{Cint},
)
    (colptr == C_NULL || rowval == C_NULL || ncolors_out == C_NULL) && return Cint(-3)
    (m > 0 && n > 0) || return Cint(-3)
    opts = _load_options(opts_ptr)
    _valid_options(opts) || return Cint(-3)
    key = combo_key(opts.structure, opts.partition, opts.decompression, opts.dtype)
    _supported_key(key) || return Cint(-2)

    # A buffer may be NULL exactly when the partition produces no coloring for
    # that dimension.
    if opts.partition == SMC_COLUMN
        column_colors == C_NULL && return Cint(-3)
    elseif opts.partition == SMC_ROW
        row_colors == C_NULL && return Cint(-3)
    else
        (row_colors == C_NULL || column_colors == C_NULL) && return Cint(-3)
    end

    base = Int(opts.index_base)
    rc = _check_pattern(Int(m), Int(n), colptr, rowval, base)
    rc == Cint(0) || return rc
    A = _build_matrix(Int(m), Int(n), colptr, rowval, base)

    order = opts.order
    pp = opts.postprocessing != Cint(0)
    sp = opts.symmetric_pattern != Cint(0)

    if key == KEY_NS_COL_DIRECT_F64 || key == KEY_NS_COL_DIRECT_F32
        color = _fast_ns_col(A, order, pp, sp)::Vector{Int64}
        return _emit_colors!(color, column_colors, Int(n), ncolors_out)
    elseif key == KEY_NS_ROW_DIRECT_F64 || key == KEY_NS_ROW_DIRECT_F32
        color = _fast_ns_row(A, order, pp, sp)::Vector{Int64}
        return _emit_colors!(color, row_colors, Int(m), ncolors_out)
    elseif key == KEY_SYM_COL_DIRECT_F64 || key == KEY_SYM_COL_DIRECT_F32
        color = _fast_sym_col_direct(A, order, pp, sp)::Vector{Int64}
        return _emit_colors!(color, column_colors, Int(n), ncolors_out)
    elseif key == KEY_SYM_COL_SUBST_F64 || key == KEY_SYM_COL_SUBST_F32
        color = _fast_sym_col_subst(A, order, pp, sp)::Vector{Int64}
        return _emit_colors!(color, column_colors, Int(n), ncolors_out)
    elseif key == KEY_NS_BID_DIRECT_F64 || key == KEY_NS_BID_DIRECT_F32
        colors = _fast_ns_bid_direct(A, order, pp, sp)::Tuple{Vector{Int64},Vector{Int64}}
        return _emit_colors2!(
            colors[1], colors[2], row_colors, column_colors, Int(m), Int(n), ncolors_out
        )
    elseif key == KEY_NS_BID_SUBST_F64 || key == KEY_NS_BID_SUBST_F32
        colors = _fast_ns_bid_subst(A, order, pp, sp)::Tuple{Vector{Int64},Vector{Int64}}
        return _emit_colors2!(
            colors[1], colors[2], row_colors, column_colors, Int(m), Int(n), ncolors_out
        )
    else
        return Cint(-2)
    end
end

# ===========================================================================
# C entry points.  Each wraps its `_do_*` helper in try/catch and returns:
#    0  success
#   -1  internal error (a Julia exception was caught and logged)
#   -2  unsupported (structure, partition, decompression, dtype) combination
#   -3  invalid argument (NULL, bad dimension, short buffer, bad enum or base)
#   -4  invalid or already-freed handle
# ===========================================================================

# Always initialise an options struct with this before overriding fields.
@export_sig smc_default_options "SmcColoringOptions"

Base.@ccallable function smc_default_options()::SmcColoringOptionsC
    return SMC_DEFAULT_OPTIONS
end

@export_sig smc_version "void" (major, "int*") (minor, "int*") (patch, "int*")

Base.@ccallable function smc_version(
    major::Ptr{Cint}, minor::Ptr{Cint}, patch::Ptr{Cint}
)::Cvoid
    major == C_NULL || unsafe_store!(major, Cint(_SMC_VERSION.major))
    minor == C_NULL || unsafe_store!(minor, Cint(_SMC_VERSION.minor))
    patch == C_NULL || unsafe_store!(patch, Cint(_SMC_VERSION.patch))
    return nothing
end

# smc_coloring — color the m-by-n CSC pattern (`colptr` / `rowval` in
# opts->index_base, NULL opts meaning the defaults) and write an opaque handle
# to `result_out`, to be released with smc_result_free.  Only the structure is
# needed; the values are passed later to smc_compress.  The caller's arrays are
# copied and never modified.
@export_sig smc_coloring "int" (m, "int") (n, "int") (colptr, "const int*") (
    rowval, "const int*"
) (opts, "const SmcColoringOptions*") (result_out, "void**")

Base.@ccallable function smc_coloring(
    m::Cint,
    n::Cint,
    colptr::Ptr{Cint},
    rowval::Ptr{Cint},
    opts::Ptr{Cvoid},
    result_out::Ptr{Ptr{Cvoid}},
)::Cint
    try
        return _do_coloring(m, n, colptr, rowval, opts, result_out)
    catch e
        @error "smc_coloring" exception = e
        return Cint(-1)
    end
end

# smc_fast_coloring — write the colors directly, without allocating a handle.
# A buffer may be NULL exactly when the partition produces no coloring for that
# dimension; SMC_BIDIRECTIONAL fills both.
@export_sig smc_fast_coloring "int" (m, "int") (n, "int") (colptr, "const int*") (
    rowval, "const int*"
) (opts, "const SmcColoringOptions*") (row_colors, "int*") (column_colors, "int*") (
    ncolors_out, "int*"
)

Base.@ccallable function smc_fast_coloring(
    m::Cint,
    n::Cint,
    colptr::Ptr{Cint},
    rowval::Ptr{Cint},
    opts::Ptr{Cvoid},
    row_colors::Ptr{Cint},
    column_colors::Ptr{Cint},
    ncolors_out::Ptr{Cint},
)::Cint
    try
        return _do_fast_coloring(
            m, n, colptr, rowval, opts, row_colors, column_colors, ncolors_out
        )
    catch e
        @error "smc_fast_coloring" exception = e
        return Cint(-1)
    end
end

# Freeing a handle twice returns -4, not a crash.
@export_sig smc_result_free "int" (result, "void*")

Base.@ccallable function smc_result_free(result::Ptr{Cvoid})::Cint
    try
        return _do_free!(result)
    catch e
        @error "smc_result_free" exception = e
        return Cint(-1)
    end
end

@export_sig smc_ncolors "int" (result, "void*") (ncolors_out, "int*")

Base.@ccallable function smc_ncolors(result::Ptr{Cvoid}, ncolors_out::Ptr{Cint})::Cint
    try
        return _do_ncolors(result, ncolors_out)
    catch e
        @error "smc_ncolors" exception = e
        return Cint(-1)
    end
end

# `len` must be at least n; -2 when the partition has no column coloring.
@export_sig smc_column_colors "int" (result, "void*") (colors, "int*") (len, "int")

Base.@ccallable function smc_column_colors(
    result::Ptr{Cvoid}, colors::Ptr{Cint}, len::Cint
)::Cint
    try
        return _do_column_colors!(result, colors, len)
    catch e
        @error "smc_column_colors" exception = e
        return Cint(-1)
    end
end

# `len` must be at least m; -2 when the partition has no row coloring.
@export_sig smc_row_colors "int" (result, "void*") (colors, "int*") (len, "int")

Base.@ccallable function smc_row_colors(
    result::Ptr{Cvoid}, colors::Ptr{Cint}, len::Cint
)::Cint
    try
        return _do_row_colors!(result, colors, len)
    catch e
        @error "smc_row_colors" exception = e
        return Cint(-1)
    end
end

# smc_ncolumn_groups / smc_nrow_groups — number of color classes.
@export_sig smc_ncolumn_groups "int" (result, "void*") (ngroups_out, "int*")

Base.@ccallable function smc_ncolumn_groups(
    result::Ptr{Cvoid}, ngroups_out::Ptr{Cint}
)::Cint
    try
        return _do_ncolumn_groups(result, ngroups_out)
    catch e
        @error "smc_ncolumn_groups" exception = e
        return Cint(-1)
    end
end

@export_sig smc_nrow_groups "int" (result, "void*") (ngroups_out, "int*")

Base.@ccallable function smc_nrow_groups(result::Ptr{Cvoid}, ngroups_out::Ptr{Cint})::Cint
    try
        return _do_nrow_groups(result, ngroups_out)
    catch e
        @error "smc_nrow_groups" exception = e
        return Cint(-1)
    end
end

# Group members.  `group` is 1-based over 1..ngroups whatever the index base;
# the member indices themselves are written in the caller's index base.
# Query the size first, then fetch.
@export_sig smc_column_group_size "int" (result, "void*") (group, "int") (size_out, "int*")

Base.@ccallable function smc_column_group_size(
    result::Ptr{Cvoid}, group::Cint, size_out::Ptr{Cint}
)::Cint
    try
        return _do_column_group_size(result, group, size_out)
    catch e
        @error "smc_column_group_size" exception = e
        return Cint(-1)
    end
end

@export_sig smc_column_group "int" (result, "void*") (group, "int") (members, "int*") (
    len, "int"
)

Base.@ccallable function smc_column_group(
    result::Ptr{Cvoid}, group::Cint, members::Ptr{Cint}, len::Cint
)::Cint
    try
        return _do_column_group!(result, group, members, len)
    catch e
        @error "smc_column_group" exception = e
        return Cint(-1)
    end
end

@export_sig smc_row_group_size "int" (result, "void*") (group, "int") (size_out, "int*")

Base.@ccallable function smc_row_group_size(
    result::Ptr{Cvoid}, group::Cint, size_out::Ptr{Cint}
)::Cint
    try
        return _do_row_group_size(result, group, size_out)
    catch e
        @error "smc_row_group_size" exception = e
        return Cint(-1)
    end
end

@export_sig smc_row_group "int" (result, "void*") (group, "int") (members, "int*") (
    len, "int"
)

Base.@ccallable function smc_row_group(
    result::Ptr{Cvoid}, group::Cint, members::Ptr{Cint}, len::Cint
)::Cint
    try
        return _do_row_group!(result, group, members, len)
    catch e
        @error "smc_row_group" exception = e
        return Cint(-1)
    end
end

# Stored entries of the colored pattern: exactly the length `nzval` must have
# in smc_compress.
@export_sig smc_nnz "int" (result, "void*") (nnz_out, "int*")

Base.@ccallable function smc_nnz(result::Ptr{Cvoid}, nnz_out::Ptr{Cint})::Cint
    try
        return _do_nnz(result, nnz_out)
    catch e
        @error "smc_nnz" exception = e
        return Cint(-1)
    end
end

# Dimensions of the colored matrix; `A_out` of smc_decompress holds m*n elements.
@export_sig smc_size "int" (result, "void*") (m_out, "int*") (n_out, "int*")

Base.@ccallable function smc_size(
    result::Ptr{Cvoid}, m_out::Ptr{Cint}, n_out::Ptr{Cint}
)::Cint
    try
        return _do_size(result, m_out, n_out)
    catch e
        @error "smc_size" exception = e
        return Cint(-1)
    end
end

# Only a bidirectional partition has a row-compressed matrix; otherwise
# *Br_rows and *Br_cols are set to 0.
@export_sig smc_compressed_size "int" (result, "void*") (Br_rows, "int*") (Br_cols, "int*") (
    Bc_rows, "int*"
) (Bc_cols, "int*")

Base.@ccallable function smc_compressed_size(
    result::Ptr{Cvoid},
    Br_rows::Ptr{Cint},
    Br_cols::Ptr{Cint},
    Bc_rows::Ptr{Cint},
    Bc_cols::Ptr{Cint},
)::Cint
    try
        return _do_compressed_size(result, Br_rows, Br_cols, Bc_rows, Bc_cols)
    catch e
        @error "smc_compressed_size" exception = e
        return Cint(-1)
    end
end

# smc_compress — compress into the dense buffers Br (row-compressed, unused and
# possibly NULL with Br_len 0 unless the partition is bidirectional) and Bc
# (column-compressed).  `nzval` holds the CSC values in the same order as the
# rowval given to smc_coloring, as double* or float* according to opts->dtype.
#
# Every `*_len` is an element count, not a byte count: nzval_len >= smc_nnz,
# Br_len >= Br_rows * Br_cols, Bc_len >= Bc_rows * Bc_cols.  Both buffers are
# column-major with the dimensions of smc_compressed_size and are written in
# full, so the caller need not zero them.  A buffer that is too small is -3,
# decided before a single element is read or written.
@export_sig smc_compress "int" (result, "void*") (nzval, "const void*") (
    nzval_len, "size_t"
) (Br, "void*") (Br_len, "size_t") (Bc, "void*") (Bc_len, "size_t")

Base.@ccallable function smc_compress(
    result::Ptr{Cvoid},
    nzval::Ptr{Cvoid},
    nzval_len::Csize_t,
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
)::Cint
    try
        return _do_compress!(result, nzval, nzval_len, Br, Br_len, Bc, Bc_len)
    catch e
        @error "smc_compress" exception = e
        return Cint(-1)
    end
end

# smc_decompress — recover the full m-by-n dense matrix (column-major); entries
# outside the sparsity pattern are set to zero.  Br / Bc follow the rules of
# smc_compress.  A_len is an element count, at least m * n from smc_size; it is
# a size_t because m * n overflows an int for perfectly ordinary dimensions.
@export_sig smc_decompress "int" (result, "void*") (Br, "const void*") (Br_len, "size_t") (
    Bc, "const void*"
) (Bc_len, "size_t") (A_out, "void*") (A_len, "size_t")

Base.@ccallable function smc_decompress(
    result::Ptr{Cvoid},
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
    A_out::Ptr{Cvoid},
    A_len::Csize_t,
)::Cint
    try
        return _do_decompress!(result, Br, Br_len, Bc, Bc_len, A_out, A_len)
    catch e
        @error "smc_decompress" exception = e
        return Cint(-1)
    end
end

end  # module LibSMC
