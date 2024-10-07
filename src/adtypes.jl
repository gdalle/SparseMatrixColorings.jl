function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::ADTypes.NoColoringAlgorithm;
    kwargs...,
)
    bg = BipartiteGraph(A)
    color = convert(Vector{Int}, ADTypes.column_coloring(A, algo))
    return ColumnColoringResult(A, bg, color)
end

function coloring(
    A::AbstractMatrix,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::ADTypes.NoColoringAlgorithm;
    kwargs...,
)
    bg = BipartiteGraph(A)
    color = convert(Vector{Int}, ADTypes.row_coloring(A, algo))
    return RowColoringResult(A, bg, color)
end
