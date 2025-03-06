function check_valid_problem(structure::Symbol, partition::Symbol)
    valid = (
        (structure == :nonsymmetric && partition in (:column, :row, :bidirectional)) ||
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
| Jacobian | mixed   | `:nonsymmetric` | `:bidirectional` | yes         |
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

    GreedyColoringAlgorithm{decompression}(order=NaturalOrder(); postprocessing=false)
    GreedyColoringAlgorithm(order=NaturalOrder(); postprocessing=false, decompression=:direct)

- `order::AbstractOrder`: the order in which the columns or rows are colored, which can impact the number of colors.
- `postprocessing::Bool`: whether or not the coloring will be refined by assigning the neutral color `0` to some vertices.
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
    postprocessing::Bool
end

function GreedyColoringAlgorithm{decompression}(
    order::AbstractOrder=NaturalOrder(); postprocessing::Bool=false
) where {decompression}
    check_valid_algorithm(decompression)
    return GreedyColoringAlgorithm{decompression,typeof(order)}(order, postprocessing)
end

function GreedyColoringAlgorithm(
    order::AbstractOrder=NaturalOrder();
    postprocessing::Bool=false,
    decompression::Symbol=:direct,
)
    check_valid_algorithm(decompression)
    return GreedyColoringAlgorithm{decompression,typeof(order)}(order, postprocessing)
end

## Coloring

abstract type WithOrWithoutResult end
struct WithResult <: WithOrWithoutResult end
struct WithoutResult <: WithOrWithoutResult end

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

julia> collect.(column_groups(result))
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
function coloring(
    A::AbstractMatrix,
    problem::ColoringProblem,
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
)
    return _coloring(
        WithResult(), A, problem, algo; decompression_eltype, symmetric_pattern
    )
end

"""
    fast_coloring(
        S::AbstractMatrix,
        problem::ColoringProblem,
        algo::GreedyColoringAlgorithm;
        [symmetric_pattern=false]
    )

Solve a [`ColoringProblem`](@ref) on the matrix `S` with a [`GreedyColoringAlgorithm`](@ref) and return

- a single color vector for `:column` and `:row` problems
- a tuple of color vectors for `:bidirectional` problems

This function is very similar to [`coloring`](@ref), but it skips the computation of an [`AbstractColoringResult`](@ref) to speed things up.

# See also

- [`coloring`](@ref)
"""
function fast_coloring(
    A::AbstractMatrix,
    problem::ColoringProblem,
    algo::GreedyColoringAlgorithm;
    symmetric_pattern::Bool=false,
)
    return _coloring(
        WithoutResult(), A, problem, algo; decompression_eltype=Float64, symmetric_pattern
    )
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type,
    symmetric_pattern::Bool,
)
    bg = BipartiteGraph(
        A; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    if speed_setting isa WithResult
        return ColumnColoringResult(A, bg, color)
    else
        return color
    end
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type,
    symmetric_pattern::Bool,
)
    bg = BipartiteGraph(
        A; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    if speed_setting isa WithResult
        return RowColoringResult(A, bg, color)
    else
        return color
    end
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:direct};
    decompression_eltype::Type,
    symmetric_pattern::Bool,
)
    ag = AdjacencyGraph(A)
    color, star_set = star_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    if speed_setting isa WithResult
        return StarSetColoringResult(A, ag, color, star_set)
    else
        return color
    end
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:substitution};
    decompression_eltype::Type{R},
    symmetric_pattern::Bool,
) where {R}
    ag = AdjacencyGraph(A)
    color, tree_set = acyclic_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    if speed_setting isa WithResult
        return TreeSetColoringResult(A, ag, color, tree_set, decompression_eltype)
    else
        return color
    end
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:bidirectional},
    algo::GreedyColoringAlgorithm{:direct};
    decompression_eltype::Type{R},
    symmetric_pattern::Bool,
) where {R}
    A_and_Aᵀ = bidirectional_pattern(A; symmetric_pattern)
    ag = AdjacencyGraph(A_and_Aᵀ; has_diagonal=false)
    color, star_set = star_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    if speed_setting isa WithResult
        symmetric_result = StarSetColoringResult(A_and_Aᵀ, ag, color, star_set)
        return BicoloringResult(A, ag, symmetric_result, decompression_eltype)
    else
        row_color, column_color, _ = remap_colors(color, maximum(color), size(A)...)
        return row_color, column_color
    end
end

function _coloring(
    speed_setting::WithOrWithoutResult,
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:bidirectional},
    algo::GreedyColoringAlgorithm{:substitution};
    decompression_eltype::Type{R},
    symmetric_pattern::Bool,
) where {R}
    A_and_Aᵀ = bidirectional_pattern(A; symmetric_pattern)
    ag = AdjacencyGraph(A_and_Aᵀ; has_diagonal=false)
    color, tree_set = acyclic_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    if speed_setting isa WithResult
        symmetric_result = TreeSetColoringResult(
            A_and_Aᵀ, ag, color, tree_set, decompression_eltype
        )
        return BicoloringResult(A, ag, symmetric_result, decompression_eltype)
    else
        row_color, column_color, _ = remap_colors(color, maximum(color), size(A)...)
        return row_color, column_color
    end
end

## ADTypes interface

function ADTypes.column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return fast_coloring(A, ColoringProblem{:nonsymmetric,:column}(), algo)
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return fast_coloring(A, ColoringProblem{:nonsymmetric,:row}(), algo)
end

function ADTypes.symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    return fast_coloring(A, ColoringProblem{:symmetric,:column}(), algo)
end
