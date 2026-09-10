# c_stores.jl — typed handle stores and every operation that dispatches on a
# handle (queries, compression, decompression).
#
# `smc_coloring` roots each result in one of the nine typed stores of
# section 3 and hands back an opaque handle; a parallel key store records the
# combo key, so a later call can recover the *concrete* type of the result,
# which is what `--trim=safe` requires.  Nine stores and not twelve because
# `decompression_eltype` is absent from the Column / Row / StarSet result types.

# The internal matrix is always Float64 / Int64: a coloring only looks at the
# structure, and `dtype` is a property of the compressed buffers, not the pattern.
const SmcMatrix = SparseMatrixCSC{Float64,Int64}

# The element type of every `group` field: a view into one contiguous block of
# a single flat index vector (see `SparseMatrixColorings.group_by_color`).
const SmcGroupView = SubArray{Int64,1,Vector{Int64},Tuple{UnitRange{Int64}},true}
const SmcGroups = Vector{SmcGroupView}

const SmcColumnResult = SparseMatrixColorings.ColumnColoringResult{
    SmcMatrix,
    Int64,
    SparseMatrixColorings.BipartiteGraph{Int64},
    Vector{Int64},
    SmcGroups,
    Vector{Int64},
    Nothing,
}

const SmcRowResult = SparseMatrixColorings.RowColoringResult{
    SmcMatrix,
    Int64,
    SparseMatrixColorings.BipartiteGraph{Int64},
    Vector{Int64},
    SmcGroups,
    Vector{Int64},
    Nothing,
}

const SmcStarResult = SparseMatrixColorings.StarSetColoringResult{
    SmcMatrix,
    Int64,
    SparseMatrixColorings.AdjacencyGraph{Int64,false},
    Vector{Int64},
    SmcGroups,
    Vector{Int64},
    Nothing,
}

const SmcTreeResult{R} = SparseMatrixColorings.TreeSetColoringResult{
    SmcMatrix,Int64,SparseMatrixColorings.AdjacencyGraph{Int64,false},SmcGroups,R
}

# A bicoloring colors the augmented matrix [0 Aᵀ; A 0], whose pattern is a
# `SparsityPatternCSC` rather than a `SparseMatrixCSC`, so its inner symmetric
# result is a *different* concrete type from the ones above.
const SmcAugStarResult = SparseMatrixColorings.StarSetColoringResult{
    SparseMatrixColorings.SparsityPatternCSC{Int64},
    Int64,
    SparseMatrixColorings.AdjacencyGraph{Int64,true},
    Vector{Int64},
    SmcGroups,
    Vector{Int64},
    Nothing,
}

const SmcAugTreeResult{R} = SparseMatrixColorings.TreeSetColoringResult{
    SparseMatrixColorings.SparsityPatternCSC{Int64},
    Int64,
    SparseMatrixColorings.AdjacencyGraph{Int64,true},
    SmcGroups,
    R,
}

const SmcBiDirectResult{R} = SparseMatrixColorings.BicoloringResult{
    SmcMatrix,
    Int64,
    SparseMatrixColorings.AdjacencyGraph{Int64,true},
    :direct,
    SmcGroups,
    SmcAugStarResult,
    R,
}

const SmcBiSubstResult{R} = SparseMatrixColorings.BicoloringResult{
    SmcMatrix,
    Int64,
    SparseMatrixColorings.AdjacencyGraph{Int64,true},
    :substitution,
    SmcGroups,
    SmcAugTreeResult{R},
    R,
}

# Shorthand used to dispatch the typed helpers on the *partition* of a result.
const SmcResult{structure,partition,decompression} = SparseMatrixColorings.AbstractColoringResult{
    structure,partition,decompression
}

# Every store value type is concrete, so `--trim=safe` can resolve every call
# made on a value fetched out of a store.

const store_ns_col_direct = Dict{Ptr{Cvoid},SmcColumnResult}()
const store_ns_row_direct = Dict{Ptr{Cvoid},SmcRowResult}()
const store_sym_col_direct = Dict{Ptr{Cvoid},SmcStarResult}()
const store_sym_col_subst_f64 = Dict{Ptr{Cvoid},SmcTreeResult{Float64}}()
const store_sym_col_subst_f32 = Dict{Ptr{Cvoid},SmcTreeResult{Float32}}()
const store_ns_bid_direct_f64 = Dict{Ptr{Cvoid},SmcBiDirectResult{Float64}}()
const store_ns_bid_direct_f32 = Dict{Ptr{Cvoid},SmcBiDirectResult{Float32}}()
const store_ns_bid_subst_f64 = Dict{Ptr{Cvoid},SmcBiSubstResult{Float64}}()
const store_ns_bid_subst_f32 = Dict{Ptr{Cvoid},SmcBiSubstResult{Float32}}()

# handle -> combo key: also the membership test.  A handle that is absent is
# invalid or already freed, which is -4 rather than a crash.
const result_key_store = Dict{Ptr{Cvoid},UInt8}()

# handle -> index base of the caller, remembered from `smc_coloring` because the
# group queries report member indices in that base and take no options.
const result_base_store = Dict{Ptr{Cvoid},Cint}()

# The handle is a monotone counter, not the `pointer_from_objref(result)` of
# Every `AbstractColoringResult` is an immutable struct and
# `pointer_from_objref` refuses those.  A counter is also safer, since a value is
# never recycled, which is what makes use-after-free and double-free return -4
# reliably.  It is only a token, never dereferenced, and starts at 1 so that a
# valid handle is never NULL.

const _handle_counter = Ref{UInt}(0)

function _new_handle()
    _handle_counter[] += UInt(1)
    return Ptr{Cvoid}(_handle_counter[])
end

function _result_key(handle::Ptr{Cvoid})
    haskey(result_key_store, handle) || return KEY_INVALID
    return result_key_store[handle]
end

function _result_base(handle::Ptr{Cvoid})
    haskey(result_base_store, handle) || return Cint(0)
    return result_base_store[handle]
end

# `T` is concrete at every call site, so nothing here is dynamically dispatched.
function _register!(
    store::Dict{Ptr{Cvoid},T}, res::T, key::UInt8, base::Cint, result_out::Ptr{Ptr{Cvoid}}
) where {T}
    handle = _new_handle()
    store[handle] = res
    result_key_store[handle] = key
    result_base_store[handle] = base
    unsafe_store!(result_out, handle)
    return Cint(0)
end

function _unregister!(store::Dict{Ptr{Cvoid},T}, handle::Ptr{Cvoid}) where {T}
    delete!(store, handle)
    return Cint(0)
end

# Key -> store routing: twelve combo keys served by nine stores.  The dispatch
# macros below expand to an if/elseif chain over the keys, so every branch
# indexes a store whose value type is statically known.  A key that matches
# nothing is an unknown handle: -4.

const _KEY_TO_STORE = (
    (:KEY_NS_COL_DIRECT_F64, :store_ns_col_direct),
    (:KEY_NS_COL_DIRECT_F32, :store_ns_col_direct),
    (:KEY_NS_ROW_DIRECT_F64, :store_ns_row_direct),
    (:KEY_NS_ROW_DIRECT_F32, :store_ns_row_direct),
    (:KEY_SYM_COL_DIRECT_F64, :store_sym_col_direct),
    (:KEY_SYM_COL_DIRECT_F32, :store_sym_col_direct),
    (:KEY_SYM_COL_SUBST_F64, :store_sym_col_subst_f64),
    (:KEY_SYM_COL_SUBST_F32, :store_sym_col_subst_f32),
    (:KEY_NS_BID_DIRECT_F64, :store_ns_bid_direct_f64),
    (:KEY_NS_BID_DIRECT_F32, :store_ns_bid_direct_f32),
    (:KEY_NS_BID_SUBST_F64, :store_ns_bid_subst_f64),
    (:KEY_NS_BID_SUBST_F32, :store_ns_bid_subst_f32),
)

function _dispatch_chain(key, handle, f, args, with_store::Bool)
    chain = :(Cint(-4))
    for i in length(_KEY_TO_STORE):-1:1
        keyname, storename = _KEY_TO_STORE[i]
        subject = with_store ? storename : Expr(:ref, storename, handle)
        call = Expr(:call, f, subject, args...)
        chain = Expr(:if, :($key == $keyname), call, chain)
    end
    return chain
end

"""
    @key_dispatch key handle f args...

Expand to `f(store[handle], args...)` for the store that `key` selects.
"""
macro key_dispatch(key, handle, f, args...)
    return esc(_dispatch_chain(key, handle, f, args, false))
end

"""
    @store_dispatch key handle f args...

Expand to `f(store, args...)` for the store that `key` selects.
"""
macro store_dispatch(key, handle, f, args...)
    return esc(_dispatch_chain(key, handle, f, args, true))
end

# Small pointer helpers.  Everything is written through `unsafe_store!` on the
# caller's buffer; the caller owns that memory, so nothing needs `GC.@preserve`.

function _store_int(out::Ptr{Cint}, value::Int)
    out == C_NULL && return Cint(-3)
    unsafe_store!(out, Cint(value))
    return Cint(0)
end

# Copy a color vector (labels in 0..ncolors, never shifted by the index base).
function _copy_colors!(color, out::Ptr{Cint}, len::Cint)
    out == C_NULL && return Cint(-3)
    n = length(color)
    Int(len) < n && return Cint(-3)
    @inbounds for i in 1:n
        unsafe_store!(out, Cint(color[i]), i)
    end
    return Cint(0)
end

function _group_size(groups, group::Cint, out::Ptr{Cint})
    out == C_NULL && return Cint(-3)
    g = Int(group)
    (g < 1 || g > length(groups)) && return Cint(-3)
    unsafe_store!(out, Cint(length(groups[g])))
    return Cint(0)
end

# Group members, written in the caller's index base.
function _copy_group!(groups, group::Cint, out::Ptr{Cint}, len::Cint, base::Cint)
    out == C_NULL && return Cint(-3)
    g = Int(group)
    (g < 1 || g > length(groups)) && return Cint(-3)
    members = groups[g]
    nm = length(members)
    Int(len) < nm && return Cint(-3)
    shift = Int(base) - 1
    @inbounds for i in 1:nm
        unsafe_store!(out, Cint(Int(members[i]) + shift), i)
    end
    return Cint(0)
end

function _zero_fill!(p::Ptr{R}, len::Int) where {R}
    @inbounds for i in 1:len
        unsafe_store!(p, zero(R), i)
    end
    return nothing
end

# A column-major view of a caller buffer.  A zero-sized compressed matrix may
# come in as NULL, in which case an empty Julia matrix stands in.
function _wrap_matrix(::Type{R}, p::Ptr{Cvoid}, nrows::Int, ncols::Int) where {R}
    (p == C_NULL || nrows * ncols == 0) && return Matrix{R}(undef, nrows, ncols)
    return unsafe_wrap(Matrix{R}, Ptr{R}(p), (nrows, ncols))
end

# Hand-rolled compression, for two independent reasons: these loops write
# straight into the caller's buffer, whereas `SparseMatrixColorings.compress`
# allocates a `Matrix` we would copy and throw away; and `compress` would not
# trim, since its `sum` over a sparse matrix bottoms out in an unresolvable
# `Base.mapreduce_empty(::typeof(identity), ::Function, T)::Any`.
#
# The result is identical: B[:, c] is the sum of the columns of group c,
# B[c, :] the sum of its rows, and the neutral color 0 contributes to nothing.

# B is nrows-by-* column-major; B[rowval[k], color[j]] += nzval[k]
function _accumulate_columns!(
    B::Ptr{R}, nzval::Ptr{R}, A::SmcMatrix, color, nrows::Int
) where {R}
    colptr = A.colptr
    rowval = A.rowval
    @inbounds for j in 1:size(A, 2)
        cj = Int(color[j])
        cj == 0 && continue
        offset = (cj - 1) * nrows
        for k in colptr[j]:(colptr[j + 1] - 1)
            idx = offset + Int(rowval[k])
            unsafe_store!(B, unsafe_load(B, idx) + unsafe_load(nzval, k), idx)
        end
    end
    return nothing
end

# B is nrows-by-* column-major; B[color[rowval[k]], j] += nzval[k]
function _accumulate_rows!(
    B::Ptr{R}, nzval::Ptr{R}, A::SmcMatrix, color, nrows::Int
) where {R}
    colptr = A.colptr
    rowval = A.rowval
    @inbounds for j in 1:size(A, 2)
        offset = (j - 1) * nrows
        for k in colptr[j]:(colptr[j + 1] - 1)
            ci = Int(color[rowval[k]])
            ci == 0 && continue
            idx = offset + ci
            unsafe_store!(B, unsafe_load(B, idx) + unsafe_load(nzval, k), idx)
        end
    end
    return nothing
end

# Typed helpers.  Each is called from a branch where the result has a statically
# known concrete type, so the partition dispatch below resolves at compile time.
# A query the partition cannot answer (row information of a column coloring, and
# vice versa) is -2, not -3: the argument is well formed, there is just no result.

_typed_ncolors(res, out::Ptr{Cint}) = _store_int(out, SparseMatrixColorings.ncolors(res))

## Column colors and groups — absent from a :row partition.

_typed_column_colors!(::SmcResult{s,:row,d}, ::Ptr{Cint}, ::Cint) where {s,d} = Cint(-2)

function _typed_column_colors!(
    res::SmcResult{s,:column,d}, out::Ptr{Cint}, len::Cint
) where {s,d}
    return _copy_colors!(SparseMatrixColorings.column_colors(res), out, len)
end

function _typed_column_colors!(
    res::SmcResult{s,:bidirectional,d}, out::Ptr{Cint}, len::Cint
) where {s,d}
    return _copy_colors!(SparseMatrixColorings.column_colors(res), out, len)
end

_typed_ncolumn_groups(::SmcResult{s,:row,d}, ::Ptr{Cint}) where {s,d} = Cint(-2)

function _typed_ncolumn_groups(res::SmcResult{s,:column,d}, out::Ptr{Cint}) where {s,d}
    return _store_int(out, length(SparseMatrixColorings.column_groups(res)))
end

function _typed_ncolumn_groups(
    res::SmcResult{s,:bidirectional,d}, out::Ptr{Cint}
) where {s,d}
    return _store_int(out, length(SparseMatrixColorings.column_groups(res)))
end

_typed_column_group_size(::SmcResult{s,:row,d}, ::Cint, ::Ptr{Cint}) where {s,d} = Cint(-2)

function _typed_column_group_size(
    res::SmcResult{s,:column,d}, group::Cint, out::Ptr{Cint}
) where {s,d}
    return _group_size(SparseMatrixColorings.column_groups(res), group, out)
end

function _typed_column_group_size(
    res::SmcResult{s,:bidirectional,d}, group::Cint, out::Ptr{Cint}
) where {s,d}
    return _group_size(SparseMatrixColorings.column_groups(res), group, out)
end

function _typed_column_group!(
    ::SmcResult{s,:row,d}, ::Cint, ::Ptr{Cint}, ::Cint, ::Cint
) where {s,d}
    return Cint(-2)
end

function _typed_column_group!(
    res::SmcResult{s,:column,d}, group::Cint, out::Ptr{Cint}, len::Cint, base::Cint
) where {s,d}
    return _copy_group!(SparseMatrixColorings.column_groups(res), group, out, len, base)
end

function _typed_column_group!(
    res::SmcResult{s,:bidirectional,d}, group::Cint, out::Ptr{Cint}, len::Cint, base::Cint
) where {s,d}
    return _copy_group!(SparseMatrixColorings.column_groups(res), group, out, len, base)
end

## Row colors and groups — absent from a :column partition.

_typed_row_colors!(::SmcResult{s,:column,d}, ::Ptr{Cint}, ::Cint) where {s,d} = Cint(-2)

function _typed_row_colors!(res::SmcResult{s,:row,d}, out::Ptr{Cint}, len::Cint) where {s,d}
    return _copy_colors!(SparseMatrixColorings.row_colors(res), out, len)
end

function _typed_row_colors!(
    res::SmcResult{s,:bidirectional,d}, out::Ptr{Cint}, len::Cint
) where {s,d}
    return _copy_colors!(SparseMatrixColorings.row_colors(res), out, len)
end

_typed_nrow_groups(::SmcResult{s,:column,d}, ::Ptr{Cint}) where {s,d} = Cint(-2)

function _typed_nrow_groups(res::SmcResult{s,:row,d}, out::Ptr{Cint}) where {s,d}
    return _store_int(out, length(SparseMatrixColorings.row_groups(res)))
end

function _typed_nrow_groups(res::SmcResult{s,:bidirectional,d}, out::Ptr{Cint}) where {s,d}
    return _store_int(out, length(SparseMatrixColorings.row_groups(res)))
end

_typed_row_group_size(::SmcResult{s,:column,d}, ::Cint, ::Ptr{Cint}) where {s,d} = Cint(-2)

function _typed_row_group_size(
    res::SmcResult{s,:row,d}, group::Cint, out::Ptr{Cint}
) where {s,d}
    return _group_size(SparseMatrixColorings.row_groups(res), group, out)
end

function _typed_row_group_size(
    res::SmcResult{s,:bidirectional,d}, group::Cint, out::Ptr{Cint}
) where {s,d}
    return _group_size(SparseMatrixColorings.row_groups(res), group, out)
end

function _typed_row_group!(
    ::SmcResult{s,:column,d}, ::Cint, ::Ptr{Cint}, ::Cint, ::Cint
) where {s,d}
    return Cint(-2)
end

function _typed_row_group!(
    res::SmcResult{s,:row,d}, group::Cint, out::Ptr{Cint}, len::Cint, base::Cint
) where {s,d}
    return _copy_group!(SparseMatrixColorings.row_groups(res), group, out, len, base)
end

function _typed_row_group!(
    res::SmcResult{s,:bidirectional,d}, group::Cint, out::Ptr{Cint}, len::Cint, base::Cint
) where {s,d}
    return _copy_group!(SparseMatrixColorings.row_groups(res), group, out, len, base)
end

## Dimensions of the compressed matrices, as `(Br_rows, Br_cols, Bc_rows, Bc_cols)`.
## Only a bidirectional partition has a row-compressed matrix.

function _compressed_dims(res::SmcResult{s,:column,d}) where {s,d}
    return (0, 0, size(res.A, 1), length(SparseMatrixColorings.column_groups(res)))
end

function _compressed_dims(res::SmcResult{s,:row,d}) where {s,d}
    return (0, 0, length(SparseMatrixColorings.row_groups(res)), size(res.A, 2))
end

function _compressed_dims(res::SmcResult{s,:bidirectional,d}) where {s,d}
    return (
        length(SparseMatrixColorings.row_groups(res)),
        size(res.A, 2),
        size(res.A, 1),
        length(SparseMatrixColorings.column_groups(res)),
    )
end

## Dispatches on the partition type parameter, so each caller of the buffer
## checks below sees a compile-time constant.

_bidirectional(::SmcResult{s,:column,d}) where {s,d} = false
_bidirectional(::SmcResult{s,:row,d}) where {s,d} = false
_bidirectional(::SmcResult{s,:bidirectional,d}) where {s,d} = true

## Sizing queries.  `res.A` is the pattern that was colored, whatever the
## partition, so both answers are partition-independent.

_typed_nnz(res, out::Ptr{Cint}) = _store_int(out, SparseArrays.nnz(res.A))

function _typed_size(res, m_out::Ptr{Cint}, n_out::Ptr{Cint})
    (m_out == C_NULL || n_out == C_NULL) && return Cint(-3)
    unsafe_store!(m_out, Cint(size(res.A, 1)))
    unsafe_store!(n_out, Cint(size(res.A, 2)))
    return Cint(0)
end

function _typed_compressed_size(
    res, br_rows::Ptr{Cint}, br_cols::Ptr{Cint}, bc_rows::Ptr{Cint}, bc_cols::Ptr{Cint}
)
    (br_rows == C_NULL || br_cols == C_NULL || bc_rows == C_NULL || bc_cols == C_NULL) &&
        return Cint(-3)
    brr, brc, bcr, bcc = _compressed_dims(res)
    unsafe_store!(br_rows, Cint(brr))
    unsafe_store!(br_cols, Cint(brc))
    unsafe_store!(bc_rows, Cint(bcr))
    unsafe_store!(bc_cols, Cint(bcc))
    return Cint(0)
end

## Buffer validation.  Lengths are ELEMENT counts of the type selected by
## `dtype`, checked here before a single element is read or written: the loops
## above and the `unsafe_wrap` of `_decompress_impl!` take the caller's word for
## the size of the buffer, so this is the only guard against a heap overrun.
## Comparisons are in unsigned `Csize_t`, so a length of 0 or of SIZE_MAX cannot
## wrap into a value that passes.

# Br is unused unless the partition is bidirectional (it may then be NULL with
# Br_len 0); a bidirectional result requires both buffers.
function _check_compressed(
    res, Br::Ptr{Cvoid}, Br_len::Csize_t, Bc::Ptr{Cvoid}, Bc_len::Csize_t
)
    brr, brc, bcr, bcc = _compressed_dims(res)
    if _bidirectional(res)
        (Br == C_NULL || Bc == C_NULL) && return Cint(-3)
        Br_len < Csize_t(brr * brc) && return Cint(-3)
    else
        (bcr * bcc > 0 && Bc == C_NULL) && return Cint(-3)
    end
    Bc_len < Csize_t(bcr * bcc) && return Cint(-3)
    return Cint(0)
end

## Compression.  `dtype` is the low bit of the combo key; the two branches make
## the element type a compile-time constant (function barrier).

function _typed_compress!(
    res,
    dtype::Cint,
    nzval::Ptr{Cvoid},
    nzval_len::Csize_t,
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
)
    nzval == C_NULL && return Cint(-3)
    nzval_len < Csize_t(SparseArrays.nnz(res.A)) && return Cint(-3)
    rc = _check_compressed(res, Br, Br_len, Bc, Bc_len)
    rc == Cint(0) || return rc
    brr, brc, bcr, bcc = _compressed_dims(res)
    if dtype == SMC_FLOAT64
        return _compress_impl!(res, Float64, nzval, Br, Bc, brr, brc, bcr, bcc)
    else
        return _compress_impl!(res, Float32, nzval, Br, Bc, brr, brc, bcr, bcc)
    end
end

function _compress_impl!(
    res::SmcResult{s,:column,d},
    ::Type{R},
    nzval::Ptr{Cvoid},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    v = Ptr{R}(nzval)
    B = Ptr{R}(Bc)
    _zero_fill!(B, bcr * bcc)
    _accumulate_columns!(B, v, res.A, SparseMatrixColorings.column_colors(res), bcr)
    return Cint(0)
end

function _compress_impl!(
    res::SmcResult{s,:row,d},
    ::Type{R},
    nzval::Ptr{Cvoid},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    v = Ptr{R}(nzval)
    B = Ptr{R}(Bc)
    _zero_fill!(B, bcr * bcc)
    _accumulate_rows!(B, v, res.A, SparseMatrixColorings.row_colors(res), bcr)
    return Cint(0)
end

function _compress_impl!(
    res::SmcResult{s,:bidirectional,d},
    ::Type{R},
    nzval::Ptr{Cvoid},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    v = Ptr{R}(nzval)
    Brp = Ptr{R}(Br)
    Bcp = Ptr{R}(Bc)
    _zero_fill!(Brp, brr * brc)
    _zero_fill!(Bcp, bcr * bcc)
    _accumulate_rows!(Brp, v, res.A, SparseMatrixColorings.row_colors(res), brr)
    _accumulate_columns!(Bcp, v, res.A, SparseMatrixColorings.column_colors(res), bcr)
    return Cint(0)
end

## Decompression.  `decompress!` is trim-friendly and is used stock.

function _typed_decompress!(
    res,
    dtype::Cint,
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
    A_out::Ptr{Cvoid},
    A_len::Csize_t,
)
    A_out == C_NULL && return Cint(-3)
    # `_decompress_impl!` wraps an m-by-n `Matrix` over A_out and that wrap
    # believes whatever size it is given, so Julia's bounds checks cannot help.
    A_len < Csize_t(size(res.A, 1) * size(res.A, 2)) && return Cint(-3)
    rc = _check_compressed(res, Br, Br_len, Bc, Bc_len)
    rc == Cint(0) || return rc
    brr, brc, bcr, bcc = _compressed_dims(res)
    if dtype == SMC_FLOAT64
        return _decompress_impl!(res, Float64, Br, Bc, A_out, brr, brc, bcr, bcc)
    else
        return _decompress_impl!(res, Float32, Br, Bc, A_out, brr, brc, bcr, bcc)
    end
end

function _decompress_impl!(
    res::SmcResult{s,:column,d},
    ::Type{R},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    A_out::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    B = _wrap_matrix(R, Bc, bcr, bcc)
    A = unsafe_wrap(Matrix{R}, Ptr{R}(A_out), (size(res.A, 1), size(res.A, 2)))
    SparseMatrixColorings.decompress!(A, B, res)
    return Cint(0)
end

function _decompress_impl!(
    res::SmcResult{s,:row,d},
    ::Type{R},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    A_out::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    B = _wrap_matrix(R, Bc, bcr, bcc)
    A = unsafe_wrap(Matrix{R}, Ptr{R}(A_out), (size(res.A, 1), size(res.A, 2)))
    SparseMatrixColorings.decompress!(A, B, res)
    return Cint(0)
end

function _decompress_impl!(
    res::SmcResult{s,:bidirectional,d},
    ::Type{R},
    Br::Ptr{Cvoid},
    Bc::Ptr{Cvoid},
    A_out::Ptr{Cvoid},
    brr::Int,
    brc::Int,
    bcr::Int,
    bcc::Int,
) where {s,d,R}
    Brm = _wrap_matrix(R, Br, brr, brc)
    Bcm = _wrap_matrix(R, Bc, bcr, bcc)
    A = unsafe_wrap(Matrix{R}, Ptr{R}(A_out), (size(res.A, 1), size(res.A, 2)))
    SparseMatrixColorings.decompress!(A, Brm, Bcm, res)
    return Cint(0)
end

# Handle dispatch — one `_do_*` per C entry point that takes a handle.  The
# handle is validated first, so a stale or bogus handle is -4 whatever the state
# of the other arguments; only then are the buffers checked (-3), and only then
# does the partition decide whether the question makes sense at all (-2).

function _do_ncolors(handle::Ptr{Cvoid}, ncolors_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_ncolors ncolors_out
end

function _do_column_colors!(handle::Ptr{Cvoid}, colors::Ptr{Cint}, len::Cint)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_column_colors! colors len
end

function _do_row_colors!(handle::Ptr{Cvoid}, colors::Ptr{Cint}, len::Cint)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_row_colors! colors len
end

function _do_ncolumn_groups(handle::Ptr{Cvoid}, ngroups_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_ncolumn_groups ngroups_out
end

function _do_nrow_groups(handle::Ptr{Cvoid}, ngroups_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_nrow_groups ngroups_out
end

function _do_column_group_size(handle::Ptr{Cvoid}, group::Cint, size_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_column_group_size group size_out
end

function _do_column_group!(handle::Ptr{Cvoid}, group::Cint, members::Ptr{Cint}, len::Cint)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    base = _result_base(handle)
    return @key_dispatch key handle _typed_column_group! group members len base
end

function _do_row_group_size(handle::Ptr{Cvoid}, group::Cint, size_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_row_group_size group size_out
end

function _do_row_group!(handle::Ptr{Cvoid}, group::Cint, members::Ptr{Cint}, len::Cint)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    base = _result_base(handle)
    return @key_dispatch key handle _typed_row_group! group members len base
end

function _do_nnz(handle::Ptr{Cvoid}, nnz_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_nnz nnz_out
end

function _do_size(handle::Ptr{Cvoid}, m_out::Ptr{Cint}, n_out::Ptr{Cint})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_size m_out n_out
end

function _do_compressed_size(
    handle::Ptr{Cvoid},
    br_rows::Ptr{Cint},
    br_cols::Ptr{Cint},
    bc_rows::Ptr{Cint},
    bc_cols::Ptr{Cint},
)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    return @key_dispatch key handle _typed_compressed_size br_rows br_cols bc_rows bc_cols
end

function _do_compress!(
    handle::Ptr{Cvoid},
    nzval::Ptr{Cvoid},
    nzval_len::Csize_t,
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    dtype = Cint(key & 0x01)   # the dtype bit of the combo key
    return @key_dispatch key handle _typed_compress! dtype nzval nzval_len Br Br_len Bc Bc_len
end

function _do_decompress!(
    handle::Ptr{Cvoid},
    Br::Ptr{Cvoid},
    Br_len::Csize_t,
    Bc::Ptr{Cvoid},
    Bc_len::Csize_t,
    A_out::Ptr{Cvoid},
    A_len::Csize_t,
)
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    dtype = Cint(key & 0x01)
    return @key_dispatch key handle _typed_decompress! dtype Br Br_len Bc Bc_len A_out A_len
end

function _do_free!(handle::Ptr{Cvoid})
    key = _result_key(handle)
    key == KEY_INVALID && return Cint(-4)
    ret = @store_dispatch key handle _unregister! handle
    delete!(result_key_store, handle)
    delete!(result_base_store, handle)
    return ret
end
