module SparseMatrixColoringsJuMPExt

using ADTypes: ADTypes
using JuMP:
    Model,
    is_solved_and_feasible,
    optimize!,
    primal_status,
    set_silent,
    set_start_value,
    value,
    @variable,
    @constraint,
    @objective
using JuMP
import MathOptInterface as MOI
using SparseMatrixColorings:
    BipartiteGraph, OptimalColoringAlgorithm, nb_vertices, neighbors, pattern, vertices

function optimal_distance2_coloring(
    bg::BipartiteGraph,
    ::Val{side},
    optimizer::O;
    silent::Bool=true,
    assert_solved::Bool=true,
) where {side,O}
    other_side = 3 - side
    n = nb_vertices(bg, Val(side))
    model = Model(optimizer)
    silent && set_silent(model)
    # one variable per vertex to color, removing some renumbering symmetries
    @variable(model, 1 <= color[i=1:n] <= i, Int)
    # one variable to count the number of distinct colors
    @variable(model, ncolors, Int)
    @constraint(model, [ncolors; color] in MOI.CountDistinct(n + 1))
    # distance-2 coloring: neighbors of the same vertex must have distinct colors
    for i in vertices(bg, Val(other_side))
        neigh = neighbors(bg, Val(other_side), i)
        @constraint(model, color[neigh] in MOI.AllDifferent(length(neigh)))
    end
    # minimize the number of distinct colors (can't use maximum because they are not necessarily numbered contiguously)
    @objective(model, Min, ncolors)
    # actual solving step where time is spent
    optimize!(model)
    if assert_solved
        # assert feasibility and optimality
        @assert is_solved_and_feasible(model)
    else
        # only assert feasibility
        @assert primal_status(model) == MOI.FEASIBLE_POINT
    end
    # native solver solutions are floating point numbers
    color_int = round.(Int, value.(color))
    # remap to 1:cmax in case they are not contiguous
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
    return optimal_distance2_coloring(
        bg, Val(2), algo.optimizer; algo.silent, algo.assert_solved
    )
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::OptimalColoringAlgorithm)
    bg = BipartiteGraph(A)
    return optimal_distance2_coloring(
        bg, Val(1), algo.optimizer; algo.silent, algo.assert_solved
    )
end

end
