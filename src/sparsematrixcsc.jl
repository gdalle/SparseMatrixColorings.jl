## Result

struct DirectSparseColoringResult{structure,partition,M} <:
       AbstractColoringResult{structure,partition,:direct,M}
    matrix::M
    color::Vector{Int}
    group::Vector{Vector{Int}}
    compressed_indices::Vector{Int}
end

function DirectSparseColoringResult{structure,partition}(
    matrix::M, color::Vector{Int}, compressed_indices::Vector{Int}
) where {structure,partition,M}
    return DirectSparseColoringResult{structure,partition,M}(
        matrix, color, group_by_color(color), compressed_indices
    )
end

get_matrix(result::DirectSparseColoringResult) = result.matrix

column_colors(result::DirectSparseColoringResult{s,:column}) where {s} = result.color
column_groups(result::DirectSparseColoringResult{s,:column}) where {s} = result.group

row_colors(result::DirectSparseColoringResult{s,:row}) where {s} = result.color
row_groups(result::DirectSparseColoringResult{s,:row}) where {s} = result.group

## Coloring

function coloring(
    S::SparseMatrixCSC,
    ::ColoringProblem{:nonsymmetric,:column,:direct},
    algo::GreedyColoringAlgorithm,
)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(2), algo.order)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[j]
        # A[i, j] = B[i, c]
        compressed_indices[k] = (c - 1) * n + i
    end
    return DirectSparseColoringResult{:nonsymmetric,:column}(S, color, compressed_indices)
end

function coloring(
    S::SparseMatrixCSC,
    ::ColoringProblem{:nonsymmetric,:row,:direct},
    algo::GreedyColoringAlgorithm,
)
    bg = bipartite_graph(S)
    color = partial_distance2_coloring(bg, Val(1), algo.order)
    n = size(S, 1)
    C = maximum(color)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        c = color[i]
        # A[i, j] = B[c, j]
        compressed_indices[k] = (j - 1) * C + c
    end
    return DirectSparseColoringResult{:nonsymmetric,:row}(S, color, compressed_indices)
end

function symmetric_coloring_detailed(
    S::SparseMatrixCSC,
    ::ColoringProblem{:symmetric,:column,:direct},
    algo::GreedyColoringAlgorithm,
)
    ag = adjacency_graph(S)
    color, star_set = star_coloring(ag, algo.order)
    n = size(S, 1)
    I, J, _ = findnz(S)
    compressed_indices = zeros(Int, nnz(S))
    for k in eachindex(I, J, compressed_indices)
        i, j = I[k], J[k]
        l, c = symmetric_coefficient(i, j, color, star_set)
        # A[i, j] = B[l, c]
        compressed_indices[k] = (c - 1) * n + l
    end
    return DirectSparseColoringResult{:symmetric,:column}(S, color, compressed_indices)
end

## Decompression

function decompress_aux!(
    A::SparseMatrixCSC{R}, B::AbstractMatrix{R}, result::DirectSparseColoringResult
) where {R<:Real}
    nzA = nonzeros(A)
    ind = result.compressed_indices
    for i in eachindex(nzA, ind)
        nzA[i] = B[ind[i]]
    end
    return A
end
