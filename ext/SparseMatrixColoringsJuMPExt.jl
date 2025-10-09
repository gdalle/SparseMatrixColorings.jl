module SparseMatrixColoringsJuMPExt

using ADTypes: ADTypes
using JuMP
using LinearAlgebra
import MathOptInterface as MOI
using SparseArrays
using SparseMatrixColorings:
    BipartiteGraph, OptimalColoringAlgorithm, nb_vertices, neighbors, pattern, vertices

function optimal_distance2_coloring(
    bg::BipartiteGraph, ::Val{side}, optimizer::Type{O}
) where {side,O<:MOI.AbstractOptimizer}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    model = Model(optimizer)
    set_silent(model)
    @variable(model, 1 <= color[i=1:n] <= i, Int)
    @variable(model, ncolors, Int)
    @constraint(model, [ncolors; color] in MOI.CountDistinct(n + 1))
    for i in vertices(bg, Val(other_side))
        neigh = neighbors(bg, Val(other_side), i)
        @constraint(model, color[neigh] in MOI.AllDifferent(length(neigh)))
    end
    @objective(model, Min, ncolors)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    return round.(Int, value.(color))
end

function ADTypes.column_coloring(A::AbstractMatrix, algo::OptimalColoringAlgorithm)
    bg = BipartiteGraph(A)
    return optimal_distance2_coloring(bg, Val(2), algo.optimizer)
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::OptimalColoringAlgorithm)
    bg = BipartiteGraph(A)
    return optimal_distance2_coloring(bg, Val(1), algo.optimizer)
end

end
