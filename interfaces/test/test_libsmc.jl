# Validates the C interface by loading LibSMC.jl as a plain Julia module and
# calling its @ccallable functions directly.  Loading the juliac-compiled
# libsmc.so from a Julia process would start a second Julia runtime and crash,
# so the compiled library is tested separately from C (interfaces/test/C/).
#
#   julia --startup-file=no --project=. interfaces/test/test_libsmc.jl

using Test
using SparseArrays
using SparseMatrixColorings

include(joinpath(@__DIR__, "..", "src", "LibSMC.jl"))
using .LibSMC

# Enums — must match interfaces/include/smc.h and interfaces/src/c_enums.jl

const SMC_FLOAT64 = Cint(0)
const SMC_FLOAT32 = Cint(1)

const SMC_NONSYMMETRIC = Cint(0)
const SMC_SYMMETRIC = Cint(1)

const SMC_COLUMN = Cint(0)
const SMC_ROW = Cint(1)
const SMC_BIDIRECTIONAL = Cint(2)

const SMC_DIRECT = Cint(0)
const SMC_SUBSTITUTION = Cint(1)

const SMC_NATURAL = Cint(0)
const SMC_LARGEST_FIRST = Cint(1)
const SMC_SMALLEST_LAST = Cint(2)
const SMC_INCIDENCE_DEGREE = Cint(3)
const SMC_DYNAMIC_LARGEST_FIRST = Cint(4)

const ALL_ORDERS = (
    SMC_NATURAL,
    SMC_LARGEST_FIRST,
    SMC_SMALLEST_LAST,
    SMC_INCIDENCE_DEGREE,
    SMC_DYNAMIC_LARGEST_FIRST,
)

# Julia counterparts, indexed by `order + 1`.  RandomOrder is excluded from v1.
const ORDER_OBJECTS = (
    NaturalOrder(), LargestFirst(), SmallestLast(), IncidenceDegree(), DynamicLargestFirst()
)

const SUPPORTED_COMBOS = (
    (SMC_NONSYMMETRIC, SMC_COLUMN, SMC_DIRECT),
    (SMC_NONSYMMETRIC, SMC_ROW, SMC_DIRECT),
    (SMC_SYMMETRIC, SMC_COLUMN, SMC_DIRECT),
    (SMC_SYMMETRIC, SMC_COLUMN, SMC_SUBSTITUTION),
    (SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT),
    (SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION),
)

const UNSUPPORTED_COMBOS = (
    (SMC_NONSYMMETRIC, SMC_COLUMN, SMC_SUBSTITUTION),
    (SMC_NONSYMMETRIC, SMC_ROW, SMC_SUBSTITUTION),
    (SMC_SYMMETRIC, SMC_ROW, SMC_DIRECT),
    (SMC_SYMMETRIC, SMC_ROW, SMC_SUBSTITUTION),
    (SMC_SYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT),
    (SMC_SYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION),
)

structure_symbol(s) = s == SMC_NONSYMMETRIC ? :nonsymmetric : :symmetric
partition_symbol(p) = p == SMC_COLUMN ? :column : (p == SMC_ROW ? :row : :bidirectional)
decompression_symbol(d) = d == SMC_DIRECT ? :direct : :substitution
value_type(dt) = dt == SMC_FLOAT64 ? Float64 : Float32

# Calling the @ccallable entry points.  `Base.@ccallable` defines ordinary Julia
# methods, so rather than hard-coding the declared pointer types we read the
# (unique) method signature and convert each argument accordingly.

function _argument_types(f)
    ms = collect(methods(f))
    length(ms) == 1 || error("expected exactly 1 method for $f, found $(length(ms))")
    return collect(Base.tuple_type_tail(ms[1].sig).parameters)
end

_raw_pointer(x::Base.RefValue{S}) where {S} = Base.unsafe_convert(Ptr{S}, x)
_raw_pointer(x::Array{S}) where {S} = pointer(x)
_raw_pointer(x::Ptr) = x

_convert_argument(::Type{P}, x) where {P<:Ptr} = convert(P, _raw_pointer(x))
_convert_argument(::Type{T}, x) where {T} = convert(T, x)::T

function ccallable_call(f, args...)
    types = _argument_types(f)
    length(types) == length(args) ||
        error("$f expects $(length(types)) arguments, got $(length(args))")
    GC.@preserve args begin
        converted = ntuple(i -> _convert_argument(types[i], args[i]), length(args))
        return f(converted...)
    end
end

# Recover the options struct type from the entry point that returns it, so this
# file does not depend on its Julia-side name.
const SmcOptions = typeof(LibSMC.smc_default_options())

const DEFAULT_OPTIONS = (
    structure=SMC_NONSYMMETRIC,
    partition=SMC_COLUMN,
    decompression=SMC_DIRECT,
    order=SMC_NATURAL,
    postprocessing=Cint(0),
    symmetric_pattern=Cint(0),
    index_base=Cint(0),
    dtype=SMC_FLOAT64,
)

"""
    options(; kwargs...)

Named tuple of the eight `SmcColoringOptions` fields, with their defaults.
"""
options(; kwargs...) = merge(DEFAULT_OPTIONS, NamedTuple(k => Cint(v) for (k, v) in kwargs))

# Positional construction: c_enums.jl pins the field *order*, not the names.
function options_struct(o)
    return SmcOptions(
        o.structure,
        o.partition,
        o.decompression,
        o.order,
        o.postprocessing,
        o.symmetric_pattern,
        o.index_base,
        o.dtype,
    )
end

# Thin wrappers mirroring the C prototypes in smc.h

"CSC arrays of `S`, as `Cint` in the requested index base."
function csc_arrays(S::SparseMatrixCSC, base::Integer)
    colptr = Cint.(S.colptr .- 1 .+ base)
    rowval = Cint.(S.rowval .- 1 .+ base)
    return colptr, rowval
end

function c_coloring(S::SparseMatrixCSC, o)
    colptr, rowval = csc_arrays(S, o.index_base)
    opts = Ref(options_struct(o))
    handle = Ref(Ptr{Cvoid}(C_NULL))
    ret = ccallable_call(
        LibSMC.smc_coloring, size(S, 1), size(S, 2), colptr, rowval, opts, handle
    )
    return Int(ret), handle[]
end

"Compute a coloring, failing the test (loudly) if the call does not succeed."
function c_coloring_ok(S::SparseMatrixCSC, o)
    ret, handle = c_coloring(S, o)
    ret == 0 || error("smc_coloring returned $ret for options $o")
    handle != C_NULL || error("smc_coloring returned a NULL handle for options $o")
    return handle
end

c_result_free(handle) = Int(ccallable_call(LibSMC.smc_result_free, handle))

function c_ncolors(handle)
    out = Ref(Cint(-1))
    ret = ccallable_call(LibSMC.smc_ncolors, handle, out)
    return Int(ret), Int(out[])
end

# At least one element, so a short `len` is tested against a non-NULL buffer.
function c_colors(f, handle, len::Integer)
    buffer = fill(Cint(-999), max(len, 1))
    ret = ccallable_call(f, handle, buffer, len)
    return Int(ret), Int.(buffer[1:max(len, 0)])
end

c_column_colors(handle, n) = c_colors(LibSMC.smc_column_colors, handle, n)
c_row_colors(handle, m) = c_colors(LibSMC.smc_row_colors, handle, m)

function c_ngroups(f, handle)
    out = Ref(Cint(-1))
    ret = ccallable_call(f, handle, out)
    return Int(ret), Int(out[])
end

c_ncolumn_groups(handle) = c_ngroups(LibSMC.smc_ncolumn_groups, handle)
c_nrow_groups(handle) = c_ngroups(LibSMC.smc_nrow_groups, handle)

function c_group_size(f, handle, group::Integer)
    out = Ref(Cint(-1))
    ret = ccallable_call(f, handle, group, out)
    return Int(ret), Int(out[])
end

function c_group(f, handle, group::Integer, len::Integer)
    buffer = fill(Cint(-999), max(len, 1))
    ret = ccallable_call(f, handle, group, buffer, len)
    return Int(ret), Int.(buffer[1:max(len, 0)])
end

"All column groups as a vector of member-index vectors (in the caller's index base)."
function c_column_groups(handle)
    ret, ngroups = c_ncolumn_groups(handle)
    ret == 0 || error("smc_ncolumn_groups returned $ret")
    return [
        begin
            rs, size = c_group_size(LibSMC.smc_column_group_size, handle, g)
            rs == 0 || error("smc_column_group_size($g) returned $rs")
            rg, members = c_group(LibSMC.smc_column_group, handle, g, size)
            rg == 0 || error("smc_column_group($g) returned $rg")
            members
        end for g in 1:ngroups
    ]
end

function c_row_groups(handle)
    ret, ngroups = c_nrow_groups(handle)
    ret == 0 || error("smc_nrow_groups returned $ret")
    return [
        begin
            rs, size = c_group_size(LibSMC.smc_row_group_size, handle, g)
            rs == 0 || error("smc_row_group_size($g) returned $rs")
            rg, members = c_group(LibSMC.smc_row_group, handle, g, size)
            rg == 0 || error("smc_row_group($g) returned $rg")
            members
        end for g in 1:ngroups
    ]
end

function c_compressed_size(handle)
    br_rows = Ref(Cint(-1))
    br_cols = Ref(Cint(-1))
    bc_rows = Ref(Cint(-1))
    bc_cols = Ref(Cint(-1))
    ret = ccallable_call(
        LibSMC.smc_compressed_size, handle, br_rows, br_cols, bc_rows, bc_cols
    )
    return Int(ret), (Int(br_rows[]), Int(br_cols[]), Int(bc_rows[]), Int(bc_cols[]))
end

"Number of stored entries of the pattern: the required `nzval_len`."
function c_nnz(handle)
    out = Ref(Cint(-1))
    ret = ccallable_call(LibSMC.smc_nnz, handle, out)
    return Int(ret), Int(out[])
end

"Dimensions of the colored matrix: `A_len` must be at least their product."
function c_size(handle)
    m_out = Ref(Cint(-1))
    n_out = Ref(Cint(-1))
    ret = ccallable_call(LibSMC.smc_size, handle, m_out, n_out)
    return Int(ret), (Int(m_out[]), Int(n_out[]))
end

# Buffers are pre-filled with a sentinel: compress and decompress write the
# *whole* dense output, so no sentinel may survive a successful call, and every
# sentinel must survive a call rejected for a short buffer.
const SENTINEL = -987.0

"""
    c_compress(handle, nzval, dims; kwargs...)

Call `smc_compress` with buffers of exactly the size `smc_compressed_size`
announced.  `nzval_len`, `Br_len` and `Bc_len` default to the true length of
their buffer and can each be understated on its own, so a short length is a lie
about a valid buffer.  `*_null` replaces a buffer by NULL.
"""
function c_compress(
    handle,
    nzval::Vector{T},
    dims;
    nzval_len=nothing,
    Br_len=nothing,
    Bc_len=nothing,
    nzval_null=false,
    Br_null=false,
    Bc_null=false,
) where {T}
    br_rows, br_cols, bc_rows, bc_cols = dims
    Br = fill(T(SENTINEL), br_rows, br_cols)
    Bc = fill(T(SENTINEL), bc_rows, bc_cols)
    nullptr = Ptr{Cvoid}(C_NULL)
    nzval_arg = nzval_null ? nullptr : nzval
    # Br is NULL with a length of 0 for every non-bidirectional partition.
    Br_arg = (Br_null || isempty(Br)) ? nullptr : Br
    Bc_arg = Bc_null ? nullptr : Bc
    ret = ccallable_call(
        LibSMC.smc_compress,
        handle,
        nzval_arg,
        something(nzval_len, length(nzval)),
        Br_arg,
        something(Br_len, length(Br)),
        Bc_arg,
        something(Bc_len, length(Bc)),
    )
    return Int(ret), Br, Bc
end

"""
    c_decompress(handle, Br, Bc, m, n; kwargs...)

Same conventions as [`c_compress`](@ref); the `A_out` buffer is always `m`-by-`n`.
"""
function c_decompress(
    handle,
    Br::Matrix{T},
    Bc::Matrix{T},
    m::Integer,
    n::Integer;
    Br_len=nothing,
    Bc_len=nothing,
    A_len=nothing,
    Br_null=false,
    Bc_null=false,
    A_null=false,
) where {T}
    A = fill(T(SENTINEL), m, n)
    nullptr = Ptr{Cvoid}(C_NULL)
    Br_arg = (Br_null || isempty(Br)) ? nullptr : Br
    Bc_arg = Bc_null ? nullptr : Bc
    A_arg = A_null ? nullptr : A
    ret = ccallable_call(
        LibSMC.smc_decompress,
        handle,
        Br_arg,
        something(Br_len, length(Br)),
        Bc_arg,
        something(Bc_len, length(Bc)),
        A_arg,
        something(A_len, length(A)),
    )
    return Int(ret), A
end

function c_fast_coloring(S::SparseMatrixCSC, o; row_buffer=true, column_buffer=true)
    m, n = size(S)
    colptr, rowval = csc_arrays(S, o.index_base)
    opts = Ref(options_struct(o))
    row_colors = fill(Cint(-999), m)
    column_colors = fill(Cint(-999), n)
    nc = Ref(Cint(-1))
    ret = ccallable_call(
        LibSMC.smc_fast_coloring,
        m,
        n,
        colptr,
        rowval,
        opts,
        row_buffer ? row_colors : Ptr{Cvoid}(C_NULL),
        column_buffer ? column_colors : Ptr{Cvoid}(C_NULL),
        nc,
    )
    return Int(ret), Int.(row_colors), Int.(column_colors), Int(nc[])
end

function c_version()
    major = Ref(Cint(-1))
    minor = Ref(Cint(-1))
    patch = Ref(Cint(-1))
    ccallable_call(LibSMC.smc_version, major, minor, patch)
    return Int(major[]), Int(minor[]), Int(patch[])
end

# Reference oracle: plain Julia SparseMatrixColorings

function reference_result(S::SparseMatrixCSC, o)
    problem = ColoringProblem{structure_symbol(o.structure),partition_symbol(o.partition)}()
    algorithm = GreedyColoringAlgorithm{decompression_symbol(o.decompression)}(
        ORDER_OBJECTS[o.order + 1]; postprocessing=(o.postprocessing != 0)
    )
    return coloring(
        S,
        problem,
        algorithm;
        decompression_eltype=value_type(o.dtype),
        symmetric_pattern=(o.symmetric_pattern != 0),
    )
end

# Structural validity, checked from first principles

"Row indices of the nonzeros of column `j` (1-based)."
column_support(S::SparseMatrixCSC, j) = Set(view(rowvals(S), nzrange(S, j)))

"""
    check_column_disjointness(S, colors)

Two columns carrying the same nonzero color must not share a nonzero row.
"""
function check_column_disjointness(S::SparseMatrixCSC, colors::Vector{Int})
    n = size(S, 2)
    @test length(colors) == n
    supports = [column_support(S, j) for j in 1:n]
    offenders = Tuple{Int,Int}[]
    for j in 1:n, k in (j + 1):n
        (colors[j] == 0 || colors[k] == 0) && continue
        if colors[j] == colors[k] && !isdisjoint(supports[j], supports[k])
            push!(offenders, (j, k))
        end
    end
    @test isempty(offenders)
end

function check_row_disjointness(S::SparseMatrixCSC, colors::Vector{Int})
    return check_column_disjointness(SparseMatrixCSC(transpose(S)), colors)
end

"Adjacency lists of the off-diagonal pattern of a (structurally symmetric) `S`."
function adjacency_lists(S::SparseMatrixCSC)
    n = size(S, 2)
    neighbours = [Int[] for _ in 1:n]
    rows = rowvals(S)
    for j in 1:n, k in nzrange(S, j)
        i = rows[k]
        i == j || push!(neighbours[j], i)
    end
    return neighbours
end

"Edges `(i, j)` with `i < j` of the off-diagonal pattern."
function edge_list(S::SparseMatrixCSC)
    edges = Tuple{Int,Int}[]
    rows = rowvals(S)
    for j in 1:size(S, 2), k in nzrange(S, j)
        i = rows[k]
        i < j && push!(edges, (i, j))
    end
    return edges
end

"""
    check_proper_coloring(S, colors)

Adjacent vertices carry different (nonzero) colors.
"""
function check_proper_coloring(S::SparseMatrixCSC, colors::Vector{Int})
    bad = Tuple{Int,Int}[]
    for (i, j) in edge_list(S)
        (colors[i] == 0 || colors[j] == 0) && continue
        colors[i] == colors[j] && push!(bad, (i, j))
    end
    @test isempty(bad)
end

"""
    check_star_coloring(S, colors)

No bicolored path on four vertices `i - j - k - l`: the structural requirement
for *direct* decompression of a symmetric matrix.
"""
function check_star_coloring(S::SparseMatrixCSC, colors::Vector{Int})
    neighbours = adjacency_lists(S)
    bad = NTuple{4,Int}[]
    for (j, k) in edge_list(S)
        for i in neighbours[j], l in neighbours[k]
            (i == k || l == j || i == l) && continue
            (colors[i] == 0 || colors[j] == 0 || colors[k] == 0 || colors[l] == 0) &&
                continue
            if colors[i] == colors[k] && colors[j] == colors[l]
                push!(bad, (i, j, k, l))
            end
        end
    end
    @test isempty(bad)
end

"Union-find root with path compression."
function _find(parent::Dict{Int,Int}, x::Int)
    root = x
    while parent[root] != root
        root = parent[root]
    end
    while parent[x] != root
        parent[x], x = root, parent[x]
    end
    return root
end

"""
    check_acyclic_coloring(S, colors)

Every subgraph induced by two colors is a forest.  This is the structural
requirement for decompression by *substitution* on a symmetric matrix.
"""
function check_acyclic_coloring(S::SparseMatrixCSC, colors::Vector{Int})
    bicolored = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for (i, j) in edge_list(S)
        (colors[i] == 0 || colors[j] == 0) && continue
        key = minmax(colors[i], colors[j])
        push!(get!(bicolored, key, Tuple{Int,Int}[]), (i, j))
    end
    cycles = Tuple{Int,Int}[]
    for (key, edges) in bicolored
        parent = Dict{Int,Int}()
        for (i, j) in edges
            get!(parent, i, i)
            get!(parent, j, j)
        end
        for (i, j) in edges
            ri, rj = _find(parent, i), _find(parent, j)
            ri == rj ? push!(cycles, key) : (parent[ri] = rj)
        end
    end
    @test isempty(cycles)
end

"""
    check_direct_recoverability(S, row_colors, column_colors)

Every nonzero `A[i, j]` can be read off the compressed matrix, either from
`Bc[i, column_colors[j]]` (when no other column of that color has a nonzero in
row `i`) or from `Br[row_colors[i], j]` (mirrored condition).  Pass
`row_colors = nothing` for a column partition, `column_colors = nothing` for a
row partition.
"""
function check_direct_recoverability(S::SparseMatrixCSC, row_colors, column_colors)
    m, n = size(S)
    A = Matrix(S)
    unrecoverable = Tuple{Int,Int}[]
    for j in 1:n, i in 1:m
        iszero(A[i, j]) && continue
        by_column = false
        if column_colors !== nothing && column_colors[j] != 0
            by_column =
                !any(
                    k -> k != j && !iszero(A[i, k]) && column_colors[k] == column_colors[j],
                    1:n,
                )
        end
        by_row = false
        if row_colors !== nothing && row_colors[i] != 0
            by_row =
                !any(k -> k != i && !iszero(A[k, j]) && row_colors[k] == row_colors[i], 1:m)
        end
        (by_column || by_row) || push!(unrecoverable, (i, j))
    end
    @test isempty(unrecoverable)
end

"""
    check_symmetric_recoverability(S, colors)

Symmetric counterpart of [`check_direct_recoverability`](@ref): with a single
column-compressed `B`, `A[i, j]` is read from `B[i, colors[j]]` when column `j`
is the only one of its color meeting row `i`, or from `B[j, colors[i]]` by
symmetry.
"""
function check_symmetric_recoverability(S::SparseMatrixCSC, colors::Vector{Int})
    n = size(S, 2)
    A = Matrix(S)
    unique_in_row(i, c, skip) =
        !any(k -> k != skip && !iszero(A[i, k]) && colors[k] == c, 1:n)
    unrecoverable = Tuple{Int,Int}[]
    for j in 1:n, i in 1:n
        iszero(A[i, j]) && continue
        by_j = colors[j] != 0 && unique_in_row(i, colors[j], j)
        by_i = colors[i] != 0 && unique_in_row(j, colors[i], i)
        (by_j || by_i) || push!(unrecoverable, (i, j))
    end
    @test isempty(unrecoverable)
end

"""
    check_groups(groups, colors, base)

The groups are exactly the fibers of the color vector: group `g` lists the
indices of color `g`, in the caller's index base.
"""
function check_groups(groups::Vector{Vector{Int}}, colors::Vector{Int}, base::Integer)
    expected = [findall(==(g), colors) .- 1 .+ base for g in 1:length(groups)]
    @test [sort(g) for g in groups] == expected
    members = reduce(vcat, groups; init=Int[])
    @test length(members) == length(unique(members))                 # disjoint
    @test sort(members) == findall(!=(0), colors) .- 1 .+ base       # and exhaustive
    @test all(0 .<= colors .<= length(groups))
end

# Test matrices (integer valued, so Float32 and Float64 arithmetic is exact)

# 4x6 rectangular, nonsymmetric (the matrix from the `compress` docstring)
const A_NONSYM = sparse(Float64[
    0 0 4 6 0 9
    1 0 0 0 7 0
    0 2 0 0 8 0
    0 3 5 0 0 0
])

# 7x5 rectangular, more nonzeros per row/column
const A_NONSYM2 = sparse(
    Float64[
        1 0 0 2 0
        3 4 0 0 5
        0 6 7 0 0
        0 0 8 9 0
        2 0 0 3 4
        0 5 0 0 6
        7 0 8 0 0
    ]
)

# 7x7 symmetric, nonzero diagonal (arrow + tridiagonal)
const A_SYM = sparse(
    Float64[
        2 1 0 0 0 0 3
        1 2 1 0 0 0 0
        0 1 2 1 0 0 0
        0 0 1 2 1 0 0
        0 0 0 1 2 1 0
        0 0 0 0 1 2 1
        3 0 0 0 0 1 2
    ],
)

# 6x6 symmetric with a *zero diagonal*, so postprocessing can assign color 0.
const A_SYM_ZERO_DIAG = sparse(
    Float64[
        0 1 1 0 0 0
        1 0 0 1 0 0
        1 0 0 0 1 0
        0 1 0 0 0 1
        0 0 1 0 0 1
        0 0 0 1 1 0
    ]
)

function matrices_for(structure)
    return structure == SMC_SYMMETRIC ? (A_SYM, A_SYM_ZERO_DIAG) : (A_NONSYM, A_NONSYM2)
end

# 1. Options: layout, defaults

function test_options()
    @test fieldcount(SmcOptions) == 8
    @test all(T -> T === Cint, fieldtypes(SmcOptions))
    @test sizeof(SmcOptions) == 8 * sizeof(Cint)
    @test isbitstype(SmcOptions)

    defaults = LibSMC.smc_default_options()
    values = [getfield(defaults, i) for i in 1:8]
    @test values ==
        Cint[SMC_NONSYMMETRIC, SMC_COLUMN, SMC_DIRECT, SMC_NATURAL, 0, 0, 0, SMC_FLOAT64]
end

function test_version()
    major, minor, patch = c_version()
    v = pkgversion(SparseMatrixColorings)
    @test (major, minor, patch) == (v.major, v.minor, v.patch)
end

# 2. Coloring: structural validity + agreement with the Julia oracle

function test_coloring(S, o)
    m, n = size(S)
    handle = c_coloring_ok(S, o)
    try
        reference = reference_result(S, o)

        ret, nc = c_ncolors(handle)
        @test ret == 0
        @test nc == ncolors(reference)

        if o.partition != SMC_ROW
            ret, colors = c_column_colors(handle, n)
            @test ret == 0
            @test colors == column_colors(reference)
            groups = c_column_groups(handle)
            check_groups(groups, colors, o.index_base)
        end
        if o.partition != SMC_COLUMN
            ret, colors = c_row_colors(handle, m)
            @test ret == 0
            @test colors == row_colors(reference)
            groups = c_row_groups(handle)
            check_groups(groups, colors, o.index_base)
        end

        # -- structural validity, from the pattern alone -------------------------
        if o.partition == SMC_COLUMN
            colors = c_column_colors(handle, n)[2]
            if o.structure == SMC_NONSYMMETRIC
                check_column_disjointness(S, colors)
                check_direct_recoverability(S, nothing, colors)
            else
                check_proper_coloring(S, colors)
                if o.decompression == SMC_DIRECT
                    check_star_coloring(S, colors)
                    check_symmetric_recoverability(S, colors)
                else
                    check_acyclic_coloring(S, colors)
                end
            end
        elseif o.partition == SMC_ROW
            colors = c_row_colors(handle, m)[2]
            check_row_disjointness(S, colors)
            check_direct_recoverability(S, colors, nothing)
        else
            rows = c_row_colors(handle, m)[2]
            columns = c_column_colors(handle, n)[2]
            @test length(rows) == m && length(columns) == n
            if o.decompression == SMC_DIRECT
                check_direct_recoverability(S, rows, columns)
            end
        end

        # -- number of colors is what the sizes say ------------------------------
        ret, dims = c_compressed_size(handle)
        @test ret == 0
        br_rows, br_cols, bc_rows, bc_cols = dims
        if o.partition == SMC_COLUMN
            @test (br_rows, br_cols) == (0, 0)
            @test (bc_rows, bc_cols) == (m, nc)
        elseif o.partition == SMC_ROW
            @test (br_rows, br_cols) == (0, 0)
            @test (bc_rows, bc_cols) == (nc, n)
        else
            @test (br_rows, br_cols) == (c_nrow_groups(handle)[2], n)
            @test (bc_rows, bc_cols) == (m, c_ncolumn_groups(handle)[2])
            @test br_rows + bc_cols == nc
        end
    finally
        @test c_result_free(handle) == 0
    end
end

# 3. compress -> decompress round trip

"""
    reference_compression(S, T, partition, row_groups, column_groups, base, dims)

The compressed matrix is the sum of the columns (resp. rows) of each group.
The groups are the ones the C API reports, so this cross-checks `smc_compress`
against `smc_column_group`.
"""
function reference_compression(
    S::SparseMatrixCSC,
    ::Type{T},
    partition,
    row_groups_,
    column_groups_,
    base::Integer,
    dims,
) where {T}
    A = Matrix{T}(S)
    br_rows, br_cols, bc_rows, bc_cols = dims
    Br = zeros(T, br_rows, br_cols)
    Bc = zeros(T, bc_rows, bc_cols)
    compress_rows!(B) =
        for (g, members) in enumerate(row_groups_), i in members
            B[g, :] .+= A[i - base + 1, :]
        end
    compress_columns!(B) =
        for (g, members) in enumerate(column_groups_), j in members
            B[:, g] .+= A[:, j - base + 1]
        end
    if partition == SMC_COLUMN
        compress_columns!(Bc)
    elseif partition == SMC_ROW
        compress_rows!(Bc)
    else
        compress_rows!(Br)
        compress_columns!(Bc)
    end
    return Br, Bc
end

function test_roundtrip(S, o)
    m, n = size(S)
    T = value_type(o.dtype)
    handle = c_coloring_ok(S, o)
    try
        ret, dims = c_compressed_size(handle)
        @test ret == 0

        nzval = Vector{T}(nonzeros(S))
        ret, Br, Bc = c_compress(handle, nzval, dims)
        @test ret == 0
        @test !any(==(T(SENTINEL)), Br)
        @test !any(==(T(SENTINEL)), Bc)

        # compress must agree with the definition, on the API's own groups.
        rgroups = o.partition == SMC_COLUMN ? nothing : c_row_groups(handle)
        cgroups = o.partition == SMC_ROW ? nothing : c_column_groups(handle)
        Br_ref, Bc_ref = reference_compression(
            S, T, o.partition, rgroups, cgroups, o.index_base, dims
        )
        @test Br == Br_ref
        @test Bc == Bc_ref

        ret, A = c_decompress(handle, Br, Bc, m, n)
        @test ret == 0
        @test !any(==(T(SENTINEL)), A)
        if o.decompression == SMC_DIRECT
            @test A == Matrix{T}(S)                    # exact: integer data
        else
            @test A ≈ Matrix{T}(S) atol = 1000 * eps(T) * maximum(abs, S)
        end
    finally
        @test c_result_free(handle) == 0
    end
end

# 3b. Sizing queries and buffer lengths.  Every buffer carries its length as an
# element count, checked before a single element is read or written; `smc_nnz`
# and `smc_size` are the only way a caller holding just a handle can size
# `nzval` and `A_out`.

function test_sizing_queries(S, o)
    m, n = size(S)
    handle = c_coloring_ok(S, o)
    nullptr = Ptr{Cvoid}(C_NULL)
    scratch = Ref(Cint(-1))
    try
        @test c_nnz(handle) == (0, nnz(S))
        @test c_size(handle) == (0, (m, n))

        # A NULL out-pointer is an invalid argument, not a request to skip.
        @test Int(ccallable_call(LibSMC.smc_nnz, handle, nullptr)) == -3
        @test Int(ccallable_call(LibSMC.smc_size, handle, nullptr, scratch)) == -3
        @test Int(ccallable_call(LibSMC.smc_size, handle, scratch, nullptr)) == -3
    finally
        @test c_result_free(handle) == 0
    end
    # A freed handle answers -4, like every other query.
    @test c_nnz(handle)[1] == -4
    @test c_size(handle)[1] == -4
end

function test_buffer_lengths(S, o)
    m, n = size(S)
    T = value_type(o.dtype)
    bidirectional = o.partition == SMC_BIDIRECTIONAL
    handle = c_coloring_ok(S, o)
    try
        ret, dims = c_compressed_size(handle)
        @test ret == 0
        br_len = dims[1] * dims[2]
        bc_len = dims[3] * dims[4]
        a_len = m * n
        nzval = Vector{T}(nonzeros(S))

        # The queries really are the required lengths.
        @test c_nnz(handle)[2] == length(nzval)
        @test c_size(handle)[2] == (m, n)

        # -- exactly the announced sizes are enough -----------------------------
        ret, Br, Bc = c_compress(
            handle, nzval, dims; nzval_len=length(nzval), Br_len=br_len, Bc_len=bc_len
        )
        @test ret == 0
        @test c_decompress(
            handle, Br, Bc, m, n; Br_len=br_len, Bc_len=bc_len, A_len=a_len
        )[1] == 0

        # -- one element short, one length at a time ----------------------------
        # Each buffer keeps its full size; only the announced length shrinks, so a
        # surviving sentinel proves the check ran before any element was written.
        untouched(B) = all(==(T(SENTINEL)), B)

        ret, Br_s, Bc_s = c_compress(handle, nzval, dims; nzval_len=length(nzval) - 1)
        @test ret == -3
        @test untouched(Br_s) && untouched(Bc_s)

        ret, Br_s, Bc_s = c_compress(handle, nzval, dims; Bc_len=bc_len - 1)
        @test ret == -3
        @test untouched(Br_s) && untouched(Bc_s)

        if bidirectional
            ret, Br_s, Bc_s = c_compress(handle, nzval, dims; Br_len=br_len - 1)
            @test ret == -3
            @test untouched(Br_s) && untouched(Bc_s)
        end

        ret, A = c_decompress(handle, Br, Bc, m, n; A_len=a_len - 1)
        @test ret == -3
        @test untouched(A)

        ret, A = c_decompress(handle, Br, Bc, m, n; Bc_len=bc_len - 1)
        @test ret == -3
        @test untouched(A)

        if bidirectional
            ret, A = c_decompress(handle, Br, Bc, m, n; Br_len=br_len - 1)
            @test ret == -3
            @test untouched(A)
        end

        # -- a length of zero is short for every buffer that is actually used ----
        @test c_compress(handle, nzval, dims; nzval_len=0)[1] == -3
        @test c_compress(handle, nzval, dims; Bc_len=0)[1] == -3
        @test c_decompress(handle, Br, Bc, m, n; A_len=0)[1] == -3
        @test c_decompress(handle, Br, Bc, m, n; Bc_len=0)[1] == -3
        if bidirectional
            @test c_compress(handle, nzval, dims; Br_len=0)[1] == -3
            @test c_decompress(handle, Br, Bc, m, n; Br_len=0)[1] == -3
        end

        # -- a huge length must not wrap into "too small" -----------------------
        # The comparisons are unsigned, so SIZE_MAX must be accepted as a generous
        # promise, and only the required elements written.
        huge = typemax(Csize_t)
        big_br = bidirectional ? huge : Csize_t(0)
        @test c_compress(
            handle, nzval, dims; nzval_len=huge, Br_len=big_br, Bc_len=huge
        )[1] == 0
        @test c_decompress(handle, Br, Bc, m, n; Br_len=big_br, Bc_len=huge, A_len=huge)[1] ==
            0

        # -- NULL buffers -------------------------------------------------------
        ret, A = c_decompress(handle, Br, Bc, m, n; A_null=true)
        @test ret == -3
        @test untouched(A)
        @test c_compress(handle, nzval, dims; nzval_null=true)[1] == -3
        @test c_compress(handle, nzval, dims; Bc_null=true)[1] == -3
        ret, A = c_decompress(handle, Br, Bc, m, n; Bc_null=true)
        @test ret == -3
        @test untouched(A)

        if bidirectional
            # Both compressed matrices are required.
            @test c_compress(handle, nzval, dims; Br_null=true, Br_len=0)[1] == -3
            ret, A = c_decompress(handle, Br, Bc, m, n; Br_null=true, Br_len=0)
            @test ret == -3
            @test untouched(A)
        else
            # Br is unused: NULL with a length of 0 is the documented call.
            @test c_compress(handle, nzval, dims; Br_null=true, Br_len=0)[1] == 0
            @test c_decompress(handle, Br, Bc, m, n; Br_null=true, Br_len=0)[1] == 0
        end
    finally
        @test c_result_free(handle) == 0
    end
end

# 4. fast_coloring

function test_fast_coloring(S, o)
    m, n = size(S)
    ret, rows, columns, nc = c_fast_coloring(S, o)
    @test ret == 0
    reference = reference_result(S, o)
    @test nc == ncolors(reference)
    o.partition == SMC_ROW || @test columns == column_colors(reference)
    o.partition == SMC_COLUMN || @test rows == row_colors(reference)

    # A buffer may be NULL exactly when the partition colors no such dimension.
    ret_no_row = c_fast_coloring(S, o; row_buffer=false)[1]
    ret_no_col = c_fast_coloring(S, o; column_buffer=false)[1]
    @test ret_no_row == (o.partition == SMC_COLUMN ? 0 : -3)
    @test ret_no_col == (o.partition == SMC_ROW ? 0 : -3)

    # A NULL ncolors_out is an invalid argument.
    colptr, rowval = csc_arrays(S, o.index_base)
    opts = Ref(options_struct(o))
    nullptr = Ptr{Cvoid}(C_NULL)
    @test Int(
        ccallable_call(
            LibSMC.smc_fast_coloring,
            m,
            n,
            colptr,
            rowval,
            opts,
            fill(Cint(0), m),
            fill(Cint(0), n),
            nullptr,
        ),
    ) == -3
end

# 5. index_base

function test_index_base(S, o)
    m, n = size(S)
    base0 = merge(o, (index_base=Cint(0),))
    base1 = merge(o, (index_base=Cint(1),))

    h0 = c_coloring_ok(S, base0)
    h1 = c_coloring_ok(S, base1)
    try
        @test c_ncolors(h0) == c_ncolors(h1)
        if o.partition != SMC_ROW
            @test c_column_colors(h0, n) == c_column_colors(h1, n)   # colors are labels
            g0 = c_column_groups(h0)
            g1 = c_column_groups(h1)
            @test [g .+ 1 for g in g0] == g1                          # members shift
        end
        if o.partition != SMC_COLUMN
            @test c_row_colors(h0, m) == c_row_colors(h1, m)
            @test [g .+ 1 for g in c_row_groups(h0)] == c_row_groups(h1)
        end

        # compression and decompression are unaffected by the index base
        _, dims = c_compressed_size(h0)
        @test c_compressed_size(h1)[2] == dims
        nzval = Vector{Float64}(nonzeros(S))
        _, Br0, Bc0 = c_compress(h0, nzval, dims)
        _, Br1, Bc1 = c_compress(h1, nzval, dims)
        @test Br0 == Br1
        @test Bc0 == Bc1
        @test c_decompress(h0, Br0, Bc0, m, n)[2] == c_decompress(h1, Br1, Bc1, m, n)[2]
    finally
        @test c_result_free(h0) == 0
        @test c_result_free(h1) == 0
    end
end

# 6. symmetric_pattern shortcut

function test_symmetric_pattern()
    S = A_SYM
    m, n = size(S)
    plain = options()
    asserted = options(; symmetric_pattern=1)
    h0 = c_coloring_ok(S, plain)
    h1 = c_coloring_ok(S, asserted)
    try
        @test c_column_colors(h0, n) == c_column_colors(h1, n)
        @test c_ncolors(h0) == c_ncolors(h1)
    finally
        @test c_result_free(h0) == 0
        @test c_result_free(h1) == 0
    end
end

# 7. Several live handles at once.  Results live in nine typed stores keyed by
# structure*16 + partition*4 + decompression*2 + dtype; keeping one handle of
# every kind alive checks that the key routes each query back to its own result.

function test_many_handles()
    live = Tuple{Ptr{Cvoid},SparseMatrixCSC,NamedTuple}[]
    for (structure, partition, decompression) in SUPPORTED_COMBOS,
        dtype in (SMC_FLOAT64, SMC_FLOAT32)

        o = options(; structure, partition, decompression, dtype)
        S = first(matrices_for(structure))
        push!(live, (c_coloring_ok(S, o), S, o))
    end
    handles = [h for (h, _, _) in live]
    @test length(unique(handles)) == length(handles)

    for (handle, S, o) in live
        @test c_ncolors(handle)[2] == ncolors(reference_result(S, o))
        _, dims = c_compressed_size(handle)
        T = value_type(o.dtype)
        ret, Br, Bc = c_compress(handle, Vector{T}(nonzeros(S)), dims)
        @test ret == 0
        @test c_decompress(handle, Br, Bc, size(S)...)[1] == 0
    end

    for handle in handles
        @test c_result_free(handle) == 0
    end
    for handle in handles
        @test c_result_free(handle) == -4
    end
end

# 8. Error paths

function test_unsupported_combos()
    S = A_SYM
    for (structure, partition, decompression) in UNSUPPORTED_COMBOS,
        dtype in (SMC_FLOAT64, SMC_FLOAT32)

        o = options(; structure, partition, decompression, dtype)
        ret, handle = c_coloring(S, o)
        @test ret == -2
        @test handle == C_NULL
        ret2, _, _, _ = c_fast_coloring(S, o)
        @test ret2 == -2
    end
end

function test_invalid_arguments()
    S = A_NONSYM
    m, n = size(S)
    colptr, rowval = csc_arrays(S, 0)
    opts = Ref(options_struct(options()))
    handle = Ref(Ptr{Cvoid}(C_NULL))
    nullptr = Ptr{Cvoid}(C_NULL)

    call(args...) = Int(ccallable_call(LibSMC.smc_coloring, args...))

    @test call(m, n, nullptr, rowval, opts, handle) == -3         # NULL colptr
    @test call(m, n, colptr, nullptr, opts, handle) == -3         # NULL rowval
    @test call(m, n, colptr, rowval, opts, nullptr) == -3         # NULL result_out
    @test call(0, n, colptr, rowval, opts, handle) == -3          # m == 0
    @test call(m, -1, colptr, rowval, opts, handle) == -3         # n < 0
    @test handle[] == C_NULL

    for bad in (
        options(; index_base=2),
        options(; index_base=-1),
        options(; structure=5),
        options(; partition=9),
        options(; decompression=7),
        options(; order=9),
        options(; order=-1),
        options(; dtype=4),
    )
        ret, h = c_coloring(S, bad)
        @test ret == -3
        @test h == C_NULL
    end

    # A NULL options pointer selects the defaults (documented in the examples).
    ret = call(m, n, colptr, rowval, nullptr, handle)
    @test ret == 0
    @test handle[] != C_NULL
    @test c_result_free(handle[]) == 0
end

function test_short_buffers()
    S = A_NONSYM
    m, n = size(S)
    handle = c_coloring_ok(S, options())
    nullptr = Ptr{Cvoid}(C_NULL)
    try
        @test c_column_colors(handle, n - 1)[1] == -3
        @test Int(ccallable_call(LibSMC.smc_column_colors, handle, nullptr, n)) == -3
        @test Int(ccallable_call(LibSMC.smc_ncolors, handle, nullptr)) == -3
        @test Int(ccallable_call(LibSMC.smc_ncolumn_groups, handle, nullptr)) == -3

        _, ngroups = c_ncolumn_groups(handle)
        _, size1 = c_group_size(LibSMC.smc_column_group_size, handle, 1)
        @test c_group(LibSMC.smc_column_group, handle, 1, size1 - 1)[1] == -3
        @test c_group_size(LibSMC.smc_column_group_size, handle, 0)[1] == -3
        @test c_group_size(LibSMC.smc_column_group_size, handle, ngroups + 1)[1] == -3
        @test c_group(LibSMC.smc_column_group, handle, ngroups + 1, size1)[1] == -3

        # A column partition has no row information (smc.h: -2, not -3).
        @test c_row_colors(handle, m)[1] == -2
        @test c_nrow_groups(handle)[1] == -2
        @test c_group_size(LibSMC.smc_row_group_size, handle, 1)[1] == -2

        # Lengths are element counts of the dtype, and follow their buffer.
        _, dims = c_compressed_size(handle)
        nzval = Vector{Float64}(nonzeros(S))
        @test Int(
            ccallable_call(LibSMC.smc_compress, handle, nullptr, 0, nullptr, 0, nullptr, 0)
        ) == -3
        _, Br, Bc = c_compress(handle, nzval, dims)
        @test Int(
            ccallable_call(
                LibSMC.smc_decompress, handle, nullptr, 0, Bc, length(Bc), nullptr, 0
            ),
        ) == -3
        @test Int(
            ccallable_call(
                LibSMC.smc_compressed_size, handle, nullptr, nullptr, nullptr, nullptr
            ),
        ) == -3
        @test Int(ccallable_call(LibSMC.smc_nnz, handle, nullptr)) == -3
    finally
        @test c_result_free(handle) == 0
    end
end

function test_invalid_handle()
    S = A_NONSYM
    m, n = size(S)
    handle = c_coloring_ok(S, options())
    _, dims = c_compressed_size(handle)
    nzval = Vector{Float64}(nonzeros(S))
    _, Br, Bc = c_compress(handle, nzval, dims)

    @test c_result_free(handle) == 0
    # Every entry point must reject the stale handle with -4 instead of crashing.
    @test c_result_free(handle) == -4                 # double free
    @test c_ncolors(handle)[1] == -4
    @test c_column_colors(handle, n)[1] == -4
    @test c_row_colors(handle, m)[1] == -4
    @test c_ncolumn_groups(handle)[1] == -4
    @test c_nrow_groups(handle)[1] == -4
    @test c_group_size(LibSMC.smc_column_group_size, handle, 1)[1] == -4
    @test c_group(LibSMC.smc_column_group, handle, 1, n)[1] == -4
    @test c_group_size(LibSMC.smc_row_group_size, handle, 1)[1] == -4
    @test c_group(LibSMC.smc_row_group, handle, 1, m)[1] == -4
    @test c_compressed_size(handle)[1] == -4
    @test c_nnz(handle)[1] == -4
    @test c_size(handle)[1] == -4
    @test c_compress(handle, nzval, dims)[1] == -4
    @test c_decompress(handle, Br, Bc, m, n)[1] == -4

    # An address that was never a handle is rejected the same way.
    bogus = Ptr{Cvoid}(UInt(0xdeadbeef0))
    @test c_result_free(bogus) == -4
    @test c_ncolors(bogus)[1] == -4
    @test c_nnz(bogus)[1] == -4
    @test c_size(bogus)[1] == -4
end

# 9. Pattern validation.  `_check_pattern` is the only input validator in the
# shim, and its rejections are reachable only through a malformed CSC pattern.
# `colptr[n+1]` is the *only* bound on how far `rowval` is read, since the API
# takes no nnz argument.

"`smc_coloring` on a raw (colptr, rowval) pair; frees the handle on success."
function c_coloring_raw(m, n, colptr::Vector{Cint}, rowval::Vector{Cint}, o)
    opts = Ref(options_struct(o))
    handle = Ref(Ptr{Cvoid}(C_NULL))
    ret = Int(ccallable_call(LibSMC.smc_coloring, m, n, colptr, rowval, opts, handle))
    created = handle[]
    ret == 0 && created != C_NULL && c_result_free(created)
    return ret, created
end

"`smc_fast_coloring` on a raw (colptr, rowval) pair."
function c_fast_coloring_raw(m, n, colptr::Vector{Cint}, rowval::Vector{Cint}, o)
    opts = Ref(options_struct(o))
    nc = Ref(Cint(-1))
    return Int(
        ccallable_call(
            LibSMC.smc_fast_coloring,
            m,
            n,
            colptr,
            rowval,
            opts,
            fill(Cint(-999), m),
            fill(Cint(-999), n),
            nc,
        ),
    )
end

"Both entry points must reject the same malformed pattern, and create no handle."
function check_rejected(m, n, colptr, rowval, o)
    ret, handle = c_coloring_raw(m, n, colptr, rowval, o)
    @test ret == -3
    @test handle == C_NULL
    @test c_fast_coloring_raw(m, n, colptr, rowval, o) == -3
end

function test_pattern_validation()
    base0 = options()
    base1 = options(; index_base=1)

    # Reference: a well-formed 3x3 diagonal pattern is accepted in either base.
    @test c_coloring_raw(3, 3, Cint[0, 1, 2, 3], Cint[0, 1, 2], base0)[1] == 0
    @test c_coloring_raw(3, 3, Cint[1, 2, 3, 4], Cint[1, 2, 3], base1)[1] == 0
    @test c_fast_coloring_raw(3, 3, Cint[0, 1, 2, 3], Cint[0, 1, 2], base0) == 0

    # (1) colptr[0] must be exactly the index base.
    check_rejected(3, 3, Cint[1, 2, 3, 4], Cint[0, 1, 2], base0)
    check_rejected(3, 3, Cint[0, 1, 2, 3], Cint[1, 2, 3], base1)

    # (2) colptr must be non-decreasing; the validator stops at the first decrease.
    check_rejected(3, 3, Cint[0, 3, 1, 3], Cint[0, 1, 2], base0)
    check_rejected(3, 3, Cint[1, 4, 2, 4], Cint[1, 2, 3], base1)

    # (3) every row index must land inside 1..m once the base is removed.
    check_rejected(3, 3, Cint[0, 1, 2, 3], Cint[0, 1, 3], base0)     # i == m
    check_rejected(3, 3, Cint[0, 1, 2, 3], Cint[0, 1, -1], base0)    # i < base
    check_rejected(3, 3, Cint[1, 2, 3, 4], Cint[1, 2, 4], base1)     # i == m+1
    check_rejected(3, 3, Cint[1, 2, 3, 4], Cint[1, 2, 0], base1)     # i < base

    # (4) a duplicate-free CSC pattern holds at most m*n entries, and a colptr[n]
    # claiming more is rejected *before* `rowval` is read: without that bound the
    # calls below would walk 10 entries off the end of a length-2 `rowval`.
    check_rejected(2, 2, Cint[0, 0, 10], Cint[0, 0], base0)
    return check_rejected(2, 2, Cint[1, 1, 11], Cint[1, 1], base1)
end

@testset "libsmc C interface" begin
    # `@export_sig` is a hand-written copy of each `@ccallable` signature and
    # generate_header.jl emits the C prototypes from it: drift there misdeclares
    # the ABI, and nothing else in the pipeline would notice.
    @testset "@export_sig matches the @ccallable methods" begin
        c_to_julia = Dict(
            "int" => Cint,
            "int*" => Ptr{Cint},
            "const int*" => Ptr{Cint},
            "size_t" => Csize_t,
            "void*" => Ptr{Cvoid},
            "const void*" => Ptr{Cvoid},
            "void**" => Ptr{Ptr{Cvoid}},
            "const SmcColoringOptions*" => Ptr{Cvoid},
        )
        for (name, _, args) in LibSMC.function_sigs
            f = getglobal(LibSMC, Symbol(name))
            ms = collect(methods(f))
            @test length(ms) == 1
            julia_args = collect(Base.unwrap_unionall(only(ms).sig).parameters)[2:end]
            @test length(julia_args) == length(args)
            if length(julia_args) == length(args)
                for (i, (argname, ctype)) in enumerate(args)
                    expected = get(c_to_julia, ctype, nothing)
                    expected === nothing && continue
                    @test julia_args[i] === expected
                end
            end
        end
    end

    @testset "options and version" begin
        test_options()
        test_version()
    end

    @testset "coloring $(structure)/$(partition)/$(decompression)" for (
            structure, partition, decompression
        ) in SUPPORTED_COMBOS

        @testset "order $order, postprocessing $post, dtype $dtype" for order in ALL_ORDERS,
            post in (0, 1),
            dtype in (SMC_FLOAT64, SMC_FLOAT32)

            o = options(;
                structure, partition, decompression, order, postprocessing=post, dtype
            )
            for S in matrices_for(structure)
                test_coloring(S, o)
                test_roundtrip(S, o)
                test_fast_coloring(S, o)
            end
        end

        @testset "index_base" begin
            o = options(; structure, partition, decompression)
            for S in matrices_for(structure)
                test_index_base(S, o)
            end
        end
    end

    @testset "sizing queries and buffer lengths" begin
        @testset "$(structure)/$(partition)/$(decompression), dtype $dtype" for (
                structure, partition, decompression
            ) in SUPPORTED_COMBOS,
            dtype in (SMC_FLOAT64, SMC_FLOAT32)

            o = options(; structure, partition, decompression, dtype)
            for S in matrices_for(structure)
                test_sizing_queries(S, o)
                test_buffer_lengths(S, o)
            end
        end
    end

    @testset "symmetric_pattern" begin
        test_symmetric_pattern()
    end

    @testset "several live handles" begin
        test_many_handles()
    end

    @testset "errors: unsupported combinations" begin
        test_unsupported_combos()
    end

    @testset "errors: invalid arguments" begin
        test_invalid_arguments()
    end

    @testset "errors: short and NULL buffers" begin
        test_short_buffers()
    end

    @testset "errors: invalid and freed handles" begin
        test_invalid_handle()
    end

    @testset "errors: malformed CSC patterns" begin
        test_pattern_validation()
    end
end
