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
function coloring end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
)
    bg = BipartiteGraph(
        A; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return ColumnColoringResult(A, bg, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    decompression_eltype::Type=Float64,
    symmetric_pattern::Bool=false,
)
    bg = BipartiteGraph(
        A; symmetric_pattern=symmetric_pattern || A isa Union{Symmetric,Hermitian}
    )
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return RowColoringResult(A, bg, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:direct};
    decompression_eltype::Type=Float64,
)
    ag = AdjacencyGraph(A)
    color, star_set = star_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    return StarSetColoringResult(A, ag, color, star_set)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:symmetric,:column},
    algo::GreedyColoringAlgorithm{:substitution};
    decompression_eltype::Type=Float64,
)
    ag = AdjacencyGraph(A)
    color, tree_set = acyclic_coloring(ag, algo.order; postprocessing=algo.postprocessing)
    return TreeSetColoringResult(A, ag, color, tree_set, decompression_eltype)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:bidirectional},
    algo::GreedyColoringAlgorithm{decompression};
    decompression_eltype::Type{R}=Float64,
    symmetric_pattern::Bool=false,
) where {decompression,R}

    # Build an AdjacencyGraph for the following matrix:
    # [ 0  Aᵀ ]
    # [ A  0  ]
    m, n = size(A)
    p = m + n
    S = A isa SparseMatrixCSC ? A : SparseMatrixCSC(A)
    nnzS = nnz(S)
    rowval = Vector{Int}(undef, 2 * nnzS)
    colptr = zeros(Int, p + 1)

    # Update rowval and colptr for the block A
    for i in 1:nnzS
        rowval[i] = S.rowval[i] + n
    end
    for j in 1:n
        colptr[j] = S.colptr[j]
    end

    # Update rowval and colptr for the block Aᵀ
    if symmetric_pattern
        # We use the sparsity pattern of A for Aᵀ
        for i in 1:nnzS
            rowval[nnzS + i] = S.rowval[i]
        end
        # m and n are identical because symmetric_pattern is true
        for j in 1:m
            colptr[n + j] = nnzS + S.colptr[j]
        end
        colptr[p + 1] = 2 * nnzS + 1
    else
        # We need to determine the sparsity pattern of Aᵀ
        # We adapt the code of transpose(SparsityPatternCSC) in graph.jl
        for k in 1:nnzS
            i = S.rowval[k]
            colptr[n + i] += 1
        end

        counter = 1
        for col in (n + 1):p
            nnz_col = colptr[col]
            colptr[col] = counter
            counter += nnz_col
        end

        for j in 1:n
            for index in S.colptr[j]:(S.colptr[j + 1] - 1)
                i = S.rowval[index]
                pos = colptr[n + i]
                rowval[nnzS + pos] = j
                colptr[n + i] += 1
            end
        end

        colptr[p + 1] = nnzS + counter
        @assert colptr[p + 1] == 2 * nnzS + 1
        for col in p:-1:(n + 2)
            colptr[col] = nnzS + colptr[col - 1]
        end
        colptr[n + 1] = nnzS + 1
    end

    # Create the SparsityPatternCSC of the augmented adjacency matrix
    A_and_Aᵀ = SparsityPatternCSC{Int}(p, p, colptr, rowval)
    ag = AdjacencyGraph(A_and_Aᵀ)

    if decompression == :direct
        color, star_set = star_coloring(ag, algo.order; postprocessing=algo.postprocessing)
        symmetric_result = StarSetColoringResult(A_and_Aᵀ, ag, color, star_set)
    else
        color, tree_set = acyclic_coloring(
            ag, algo.order; postprocessing=algo.postprocessing
        )
        symmetric_result = TreeSetColoringResult(
            A_and_Aᵀ, ag, color, tree_set, decompression_eltype
        )
    end
    return BicoloringResult(A, ag, symmetric_result, decompression_eltype)
end

## ADTypes interface

function ADTypes.column_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = BipartiteGraph(A; symmetric_pattern=A isa Union{Symmetric,Hermitian})
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    return color
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    bg = BipartiteGraph(A; symmetric_pattern=A isa Union{Symmetric,Hermitian})
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    return color
end

function ADTypes.symmetric_coloring(A::AbstractMatrix, algo::GreedyColoringAlgorithm)
    ag = AdjacencyGraph(A)
    # never postprocess because end users do not expect zeros
    color, star_set = star_coloring(ag, algo.order; postprocessing=false)
    return color
end
