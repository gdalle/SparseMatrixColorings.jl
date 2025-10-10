"""
    ConstantColoringAlgorithm{partition}  <: ADTypes.AbstractColoringAlgorithm

Coloring algorithm which always returns the same precomputed vector of colors.
Useful when the optimal coloring of a matrix can be determined a priori due to its specific structure (e.g. banded).

It is passed as an argument to the main function [`coloring`](@ref), but will only work if the associated `problem` has a `:column` or `:row` partition.

# Constructors

    ConstantColoringAlgorithm{partition}(matrix_template, color)
    ConstantColoringAlgorithm{partition,structure}(matrix_template, color)
    ConstantColoringAlgorithm(
        matrix_template, color;
        structure=:nonsymmetric, partition=:column
    )

- `partition::Symbol`: either `:row` or `:column`.
- `structure::Symbol`: either `:nonsymmetric` or `:symmetric`.
- `matrix_template::AbstractMatrix`: matrix for which the vector of colors was precomputed (the algorithm will only accept matrices of the exact same size).
- `color::Vector{<:Integer}`: vector of integer colors, one for each row or column (depending on `partition`).

!!! warning
    The constructor based on keyword arguments is type-unstable if these arguments are not compile-time constants.

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
- [`ADTypes.symmetric_coloring`](@extref ADTypes.symmetric_coloring)
"""
struct ConstantColoringAlgorithm{partition,structure,M<:AbstractMatrix,T<:Integer} <:
       ADTypes.AbstractColoringAlgorithm
    matrix_template::M
    color::Vector{T}

    function ConstantColoringAlgorithm{partition,structure}(
        matrix_template::AbstractMatrix, color::Vector{<:Integer}
    ) where {partition,structure}
        check_valid_problem(structure, partition)
        return new{partition,structure,typeof(matrix_template),eltype(color)}(
            matrix_template, color
        )
    end
end

function ConstantColoringAlgorithm{partition}(
    matrix_template::AbstractMatrix, color::Vector{<:Integer}
) where {partition}
    return ConstantColoringAlgorithm{partition,:nonsymmetric}(matrix_template, color)
end

function ConstantColoringAlgorithm(
    matrix_template::AbstractMatrix,
    color::Vector{<:Integer};
    structure::Symbol=:nonsymmetric,
    partition::Symbol=:column,
)
    return ConstantColoringAlgorithm{partition,structure}(matrix_template, color)
end

function check_template(algo::ConstantColoringAlgorithm, A::AbstractMatrix)
    (; matrix_template) = algo
    if size(A) != size(matrix_template)
        throw(
            DimensionMismatch(
                "`ConstantColoringAlgorithm` expected matrix of size $(size(matrix_template)) but got matrix of size $(size(A))",
            ),
        )
    end
end

function ADTypes.column_coloring(
    A::AbstractMatrix, algo::ConstantColoringAlgorithm{:column,:nonsymmetric}
)
    check_template(algo, A)
    return algo.color
end

function ADTypes.row_coloring(
    A::AbstractMatrix, algo::ConstantColoringAlgorithm{:row,:nonsymmetric}
)
    check_template(algo, A)
    return algo.color
end

function ADTypes.symmetric_coloring(
    A::AbstractMatrix, algo::ConstantColoringAlgorithm{:column,:symmetric}
)
    check_template(algo, A)
    return algo.color
end

# TODO: handle bidirectional once https://github.com/SciML/ADTypes.jl/issues/69 is done
