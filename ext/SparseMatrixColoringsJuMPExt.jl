module SparseMatrixColoringsJuMPExt

using ADTypes: ADTypes
using JuMP:
    Model,
    assert_is_solved_and_feasible,
    optimize!,
    set_silent,
    value,
    @variable,
    @constraint,
    @objective
import MathOptInterface as MOI
using SparseMatrixColorings:
    BipartiteGraph, OptimalColoringAlgorithm, nb_vertices, neighbors, pattern, vertices

function optimal_distance2_coloring(
    bg::BipartiteGraph, ::Val{side}, optimizer::O; silent::Bool=true
) where {side,O}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    model = Model(optimizer)
    silent && set_silent(model)
    @variable(model, 1 <= color[i = 1:n] <= i, Int)
    @variable(model, ncolors, Int)
    @constraint(model, [ncolors; color] in MOI.CountDistinct(n + 1))
    for i in vertices(bg, Val(other_side))
        neigh = neighbors(bg, Val(other_side), i)
        @constraint(model, color[neigh] in MOI.AllDifferent(length(neigh)))
    end
    @objective(model, Min, ncolors)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    color_int = round.(Int, value.(color))
    # remap to 1:cmax
    true_ncolors = 0
    remap = fill(0, maximum(color_int))
    for c in color_int
        if remap[c] == 0
            true_ncolors += 1
            remap[c] = true_ncolors
        end
    end
    return remap[color_int]
end

function ADTypes.column_coloring(A::AbstractMatrix, algo::OptimalColoringAlgorithm)
    bg = BipartiteGraph(A)
    return optimal_distance2_coloring(bg, Val(2), algo.optimizer; algo.silent)
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::OptimalColoringAlgorithm)
    bg = BipartiteGraph(A)
    return optimal_distance2_coloring(bg, Val(1), algo.optimizer; algo.silent)
end

end
