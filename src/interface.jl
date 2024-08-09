"""
    ColoringProblem{
        structure,
        partition,
        decompression
    }

Selector type for the coloring problem to solve, enabling multiple dispatch.

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

Greedy coloring algorithm for sparse Jacobians and Hessians, with configurable vertex order.

Compatible with the [ADTypes.jl coloring framework](https://sciml.github.io/ADTypes.jl/stable/#Coloring-algorithm).

# Constructor

    GreedyColoringAlgorithm(order::AbstractOrder=NaturalOrder())

# See also

- [`AbstractOrder`](@ref)
- [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm)
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

Solve the coloring `problem` on matrix `S` with `algo` and return an [`AbstractColoringResult`](@ref).

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays


julia> S = sparse([
           0 0 1 1 0
           1 0 0 0 1
           0 1 1 0 0
           0 1 1 0 1
       ]);

julia> problem = ColoringProblem(structure=:nonsymmetric, partition=:column);

julia> algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst());

julia> result = coloring(S, problem, algo);

julia> column_colors(result)
5-element Vector{Int64}:
 1
 2
 1
 2
 3

julia> column_groups(result)
3-element Vector{Vector{Int64}}:
 [1, 3]
 [2, 4]
 [5]
```

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

function coloring(
    S::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column,:substitution},
    algo::GreedyColoringAlgorithm,
)
    ag = adjacency_graph(S)
    color, tree_set = acyclic_coloring(ag, algo.order)
    # TODO: handle tree_set
    return DefaultColoringResult{:symmetric,:column,:substitution}(S, color)
end

## ADTypes interface

"""
    ADTypes.column_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the columns in the bipartite graph of the matrix `S`, return a vector of integer colors.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst());

julia> S = sparse([
           0 0 1 1 0
           1 0 0 0 1
           0 1 1 0 0
           0 1 1 0 1
       ]);

julia> column_coloring(S, algo)
5-element Vector{Int64}:
 1
 2
 1
 2
 3
```
"""
function ADTypes.column_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return color
end

"""
    ADTypes.row_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a partial distance-2 coloring of the rows in the bipartite graph of the matrix `S`, return a vector of integer colors.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst());

julia> S = sparse([
           0 0 1 1 0
           1 0 0 0 1
           0 1 1 0 0
           0 1 1 0 1
       ]);

julia> row_coloring(S, algo)
4-element Vector{Int64}:
 2
 2
 3
 1
```
"""
function ADTypes.row_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return color
end

"""
    ADTypes.symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)

Compute a star coloring of the columns in the adjacency graph of the symmetric matrix `S`, return a vector of integer colors.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays

julia> algo = GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst());

julia> S = sparse([
           1 1 1 1
           1 1 0 0
           1 0 1 0
           1 0 0 1
       ]);

julia> symmetric_coloring(S, algo)
4-element Vector{Int64}:
 1
 2
 2
 2
```
"""
function ADTypes.symmetric_coloring(S::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    return color
end
