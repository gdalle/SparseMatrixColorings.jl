"""
    ConstantColoringAlgorithm{partition}  <: ADTypes.AbstractColoringAlgorithm

Coloring algorithm which always returns the same precomputed vector of colors.
Useful when the optimal coloring of a matrix can be determined a priori due to its specific structure (e.g. banded).

It is passed as an argument to the main function [`coloring`](@ref), but will only work if the associated `problem` has `:nonsymmetric` structure.
Indeed, for symmetric coloring problems, we need more than just the vector of colors to allow fast decompression.

# Constructors

    ConstantColoringAlgorithm{partition}(matrix_template, color)
    ConstantColoringAlgorithm(matrix_template, color; partition=:column)

- `partition::Symbol`: either `:row` or `:column`.
- `matrix_template::AbstractMatrix`: matrix for which the vector of colors was precomputed (the algorithm will only accept matrices of the exact same size).
- `color::Vector{Int}`: vector of integer colors, one for each row or column (depending on `partition`).

!!! warning
    The second constructor (based on keyword arguments) is type-unstable.

We do not necessarily verify consistency between the matrix template and the vector of colors, this is the responsibility of the user.

# Example

```jldoctest
julia> using SparseMatrixColorings, LinearAlgebra

julia> matrix_template = Diagonal(ones(Bool, 5))
5×5 Diagonal{Bool, Vector{Bool}}:
 1  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  1

julia> color = ones(Int, 5)  # coloring a Diagonal is trivial
5-element Vector{Int64}:
 1
 1
 1
 1
 1

julia> problem = ColoringProblem(; structure=:nonsymmetric, partition=:column);

julia> algo = ConstantColoringAlgorithm(matrix_template, color; partition=:column);

julia> result = coloring(similar(matrix_template), problem, algo);

julia> column_colors(result)
5-element Vector{Int64}:
 1
 1
 1
 1
 1
```

# ADTypes coloring interface

`ConstantColoringAlgorithm` is a subtype of [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm), which means the following methods are also applicable (although they will error if the kind of coloring demanded not consistent):

- [`ADTypes.column_coloring`](@extref ADTypes.column_coloring)
- [`ADTypes.row_coloring`](@extref ADTypes.row_coloring)
"""
struct ConstantColoringAlgorithm{
    partition,M<:AbstractMatrix,R<:AbstractColoringResult{:nonsymmetric,partition,:direct}
} <: ADTypes.AbstractColoringAlgorithm
    matrix_template::M
    color::Vector{Int}
    result::R
end

function ConstantColoringAlgorithm{:column}(
    matrix_template::AbstractMatrix, color::Vector{Int}
)
    bg = BipartiteGraph(matrix_template)
    result = ColumnColoringResult(matrix_template, bg, color)
    M, R = typeof(matrix_template), typeof(result)
    return ConstantColoringAlgorithm{:column,M,R}(matrix_template, color, result)
end

function ConstantColoringAlgorithm{:row}(
    matrix_template::AbstractMatrix, color::Vector{Int}
)
    bg = BipartiteGraph(matrix_template)
    result = RowColoringResult(matrix_template, bg, color)
    M, R = typeof(matrix_template), typeof(result)
    return ConstantColoringAlgorithm{:row,M,R}(matrix_template, color, result)
end

function ConstantColoringAlgorithm(
    matrix_template::AbstractMatrix, color::Vector{Int}; partition=:column
)
    return ConstantColoringAlgorithm{partition}(matrix_template, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,partition},
    algo::ConstantColoringAlgorithm{partition};
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
) where {partition}
    (; matrix_template, result) = algo
    if size(A) != size(matrix_template)
        throw(
            DimensionMismatch(
                "`ConstantColoringAlgorithm` expected matrix of size $(size(matrix_template)) but got matrix of size $(size(A))",
            ),
        )
    else
        return result
    end
end

function ADTypes.column_coloring(
    A::AbstractMatrix, algo::ConstantColoringAlgorithm{:column}
)
    problem = ColoringProblem{:nonsymmetric,:column}()
    result = coloring(A, problem, algo)
    return column_colors(result)
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::ConstantColoringAlgorithm)
    problem = ColoringProblem{:nonsymmetric,:row}()
    result = coloring(A, problem, algo)
    return row_colors(result)
end
