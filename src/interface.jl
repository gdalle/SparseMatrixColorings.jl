function check_valid_problem(structure::Symbol, partition::Symbol)
    valid = (
        (structure == :nonsymmetric && partition in (:column, :row)) ||
        (structure == :symmetric && partition == :column)
    )
    if !valid
        throw(
            ArgumentError(
                "The combination `($(repr(structure)), $(repr(partition)))` is not supported by `ColoringProblem`.",
            ),
        )
    end
end

function check_valid_algorithm(decompression::Symbol)
    valid = decompression in (:direct, :substitution)
    if !valid
        throw(
            ArgumentError(
                "The setting `decompression=$(repr(decompression))` is not supported by `GreedyColoringAlgorithm`.",
            ),
        )
    end
end

"""
    ColoringProblem{structure,partition}

Selector type for the coloring problem to solve, enabling multiple dispatch.

It is passed as an argument to the main function [`coloring`](@ref).

# Constructors

    ColoringProblem{structure,partition}()
    ColoringProblem(; structure=:nonsymmetric, partition=:column)

- `structure::Symbol`: either `:nonsymmetric` or `:symmetric`
- `partition::Symbol`: either `:column`, `:row` or `:bidirectional`

!!! warning
    The second constructor (based on keyword arguments) is type-unstable.

#  Link to automatic differentiation

Matrix coloring is often used in automatic differentiation, and here is the translation guide:

| matrix   | mode    | `structure`     | `partition`      | implemented |
| -------- | ------- | --------------- | ---------------- | ----------- |
| Jacobian | forward | `:nonsymmetric` | `:column`        | yes         |
| Jacobian | reverse | `:nonsymmetric` | `:row`           | yes         |
| Jacobian | mixed   | `:nonsymmetric` | `:bidirectional` | no          |
| Hessian  | -       | `:symmetric`    | `:column`        | yes         |
| Hessian  | -       | `:symmetric`    | `:row`           | no          |
"""
struct ColoringProblem{structure,partition} end

function ColoringProblem(; structure::Symbol=:nonsymmetric, partition::Symbol=:column)
    check_valid_problem(structure, partition)
    return ColoringProblem{structure,partition}()
end

"""
    GreedyColoringAlgorithm{decompression} <: ADTypes.AbstractColoringAlgorithm

Greedy coloring algorithm for sparse matrices which colors columns or rows one after the other, following a configurable order.

It is passed as an argument to the main function [`coloring`](@ref).

# Constructors

    GreedyColoringAlgorithm{decompression}(order=NaturalOrder())
    GreedyColoringAlgorithm(order=NaturalOrder(); decompression=:direct)

- `order::AbstractOrder`: the order in which the columns or rows are colored, which can impact the number of colors.
- `decompression::Symbol`: either `:direct` or `:substitution`. Usually `:substitution` leads to fewer colors, at the cost of a more expensive coloring (and decompression). When `:substitution` is not applicable, it falls back on `:direct` decompression.

!!! warning
    The second constructor (based on keyword arguments) is type-unstable.

# ADTypes coloring interface

`GreedyColoringAlgorithm` is a subtype of [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm), which means the following methods are also applicable:

- [`ADTypes.column_coloring`](@extref ADTypes.column_coloring)
- [`ADTypes.row_coloring`](@extref ADTypes.row_coloring)
- [`ADTypes.symmetric_coloring`](@extref ADTypes.symmetric_coloring)

See their respective docstrings for details.

# See also

- [`AbstractOrder`](@ref)
- [`decompress`](@ref)
"""
struct GreedyColoringAlgorithm{decompression,O<:AbstractOrder} <:
       ADTypes.AbstractColoringAlgorithm
    order::O
end

function GreedyColoringAlgorithm{decompression}(
    order::AbstractOrder=NaturalOrder()
) where {decompression}
    check_valid_algorithm(decompression)
    return GreedyColoringAlgorithm{decompression,typeof(order)}(order)
end

function GreedyColoringAlgorithm(
    order::AbstractOrder=NaturalOrder(); decompression::Symbol=:direct
)
    check_valid_algorithm(decompression)
    return GreedyColoringAlgorithm{decompression,typeof(order)}(order)
end

"""
    coloring(
        S::AbstractMatrix,
        problem::ColoringProblem,
        algo::GreedyColoringAlgorithm;
        [decompression_eltype=Float64, symmetric_pattern=false]
    )

Solve a [`ColoringProblem`](@ref) on the matrix `S` with a [`GreedyColoringAlgorithm`](@ref) and return an [`AbstractColoringResult`](@ref).

The result can be used to [`compress`](@ref) and [`decompress`](@ref) a matrix `A` with the same sparsity pattern as `S`.
If `eltype(A) == decompression_eltype`, decompression might be faster.

For a `:nonsymmetric` problem (and only then), setting `symmetric_pattern=true` indicates that the pattern of nonzeros is symmetric.
This condition is weaker than the symmetry of actual values, so it can happen for some Jacobians.
Specifying it allows faster construction of the bipartite graph.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> S = sparse([
           0 0 1 1 0 1
           1 0 0 0 1 0
           0 1 0 0 1 0
           0 1 1 0 0 0
       ]);

julia> problem = ColoringProblem(; structure=:nonsymmetric, partition=:column);

julia> algo = GreedyColoringAlgorithm(; decompression=:direct);

julia> result = coloring(S, problem, algo);

julia> column_colors(result)
6-element Vector{Int64}:
 1
 1
 2
 1
 2
 3

julia> column_groups(result)
3-element Vector{Vector{Int64}}:
 [1, 2, 4]
 [3, 5]
 [6]
```

# See also

- [`ColoringProblem`](@ref)
- [`GreedyColoringAlgorithm`](@ref)
- [`AbstractColoringResult`](@ref)
- [`compress`](@ref)
- [`decompress`](@ref)
"""
function coloring end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
)
    S = convert(SparseMatrixCSC, A)
    bg = bipartite_graph(
        S; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return ColumnColoringResult(S, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
)
    S = convert(SparseMatrixCSC, A)
    bg = bipartite_graph(
        S; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return RowColoringResult(S, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:direct};
    decompression_eltype::Type=Float64,
)
    S = convert(SparseMatrixCSC, A)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    return StarSetColoringResult(S, color, star_set)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:substitution};
    decompression_eltype::Type=Float64,
)
    S = convert(SparseMatrixCSC, A)
    ag = adjacency_graph(S)
    color, tree_set = acyclic_coloring(ag, algo.order)
    return TreeSetColoringResult(S, color, tree_set, decompression_eltype)
end

## ADTypes interface

function ADTypes.column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    S = convert(SparseMatrixCSC, A)
    bg = bipartite_graph(S; symmetric_pattern=A isa Union{Symmetric,Hermitian})
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return color
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    S = convert(SparseMatrixCSC, A)
    bg = bipartite_graph(S; symmetric_pattern=A isa Union{Symmetric,Hermitian})
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return color
end

function ADTypes.symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    S = convert(SparseMatrixCSC, A)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    return color
end
