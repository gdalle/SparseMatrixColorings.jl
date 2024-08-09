"""
    ColoringProblem{structure,partition,decompression}

Selector type for the coloring problem to solve, enabling multiple dispatch.

It is used inside the main function [`coloring`](@ref).

# Constructor

    ColoringProblem(;
        structure::Symbol=:nonsymmetric,
        partition::Symbol=:column,
        decompression::Symbol=:direct,
    )

# Type parameters

- `structure::Symbol`: either `:nonsymmetric` or `:symmetric`
- `partition::Symbol`: either `:column`, `:row` or `:bidirectional`
- `decompression::Symbol`: either `:direct` or `:substitution`

#  Link to automatic differentiation

Matrix coloring is often used in automatic differentiation, and here is the translation guide:

| matrix   | mode                 | `structure`     | `partition` |
| -------- | -------------------- | --------------- | ----------- |
| Jacobian | forward              | `:nonsymmetric` | `:column`   |
| Jacobian | reverse              | `:nonsymmetric` | `:row`      |
| Hessian  | forward-over-reverse | `:symmetric`    | `:column`   |
"""
struct ColoringProblem{structure,partition,decompression} end

function ColoringProblem(;
    structure::Symbol=:nonsymmetric,
    partition::Symbol=:column,
    decompression::Symbol=:direct,
)
    @assert structure in (:nonsymmetric, :symmetric)
    @assert partition in (:column, :row, :bidirectional)
    @assert decompression in (:direct, :substitution)
    return ColoringProblem{structure,partition,decompression}()
end

"""
    GreedyColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Greedy coloring algorithm for sparse matrices, which colors columns or rows one after the other, following a configurable order.

It is used inside the main function [`coloring`](@ref).

# Constructor

    GreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())

The choice of [`AbstractOrder`](@ref) impacts the resulting number of colors.
It defaults to [`NaturalOrder`](@ref) for reproducibility, but [`LargestFirst`](@ref) can sometimes be a better option.

# ADTypes coloring interface

`GreedyColoringAlgorithm` is a subtype of [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm), which means the following methods are also applicable:

- [`ADTypes.column_coloring`](@extref ADTypes.column_coloring)
- [`ADTypes.row_coloring`](@extref ADTypes.row_coloring)
- [`ADTypes.symmetric_coloring`](@extref ADTypes.symmetric_coloring)
"""
struct GreedyColoringAlgorithm{O<:AbstractOrder} <: ADTypes.AbstractColoringAlgorithm
    order::O
end

GreedyColoringAlgorithm() = GreedyColoringAlgorithm(NaturalOrder())

"""
    coloring(
        S::AbstractMatrix,
        problem::ColoringProblem,
        algo::GreedyColoringAlgorithm
    )

Solve a [`ColoringProblem`](@ref) on the matrix `S` with a [`GreedyColoringAlgorithm`](@ref) and return an [`AbstractColoringResult`](@ref).

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> S = sparse([
           0 0 1 1 0 1
           1 0 0 0 1 0
           0 1 0 0 1 0
           0 1 1 0 0 0
       ]);

julia> problem = ColoringProblem(structure=:nonsymmetric, partition=:column);

julia> algo = GreedyColoringAlgorithm();

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

# See also

- [`ColoringProblem`](@ref)
- [`GreedyColoringAlgorithm`](@ref)
- [`AbstractColoringResult`](@ref)
"""
function coloring end

function coloring(
    S::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:column,:direct},
    algo::GreedyColoringAlgorithm,
)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return DefaultColoringResult{:nonsymmetric,:column,:direct}(S, color)
end

function coloring(
    S::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:row,:direct},
    algo::GreedyColoringAlgorithm,
)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return DefaultColoringResult{:nonsymmetric,:row,:direct}(S, color)
end

function coloring(
    S::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column,:direct},
    algo::GreedyColoringAlgorithm,
)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    # TODO: handle star_set
    return DefaultColoringResult{:symmetric,:column,:direct}(S, color)
end

## ADTypes interface

function ADTypes.column_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return color
end

function ADTypes.row_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return color
end

function ADTypes.symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    return color
end
