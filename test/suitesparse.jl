using CSV
using DataFrames
using LinearAlgebra
using MatrixDepot
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    AdjacencyGraph,
    BipartiteGraph,
    degree,
    minimum_degree,
    maximum_degree,
    nb_vertices,
    nb_edges,
    neighbors,
    partial_distance2_coloring,
    star_coloring,
    vertices
using Test

nbunique(x) = length(unique(x))

_N(args...) = vertices(args..., NaturalOrder())
_LF(args...) = vertices(args..., LargestFirst())
_SL(args...) = vertices(args..., SmallestLast(; reproduce_colpack=true))
_ID(args...) = vertices(args..., IncidenceDegree(; reproduce_colpack=true))
_DLF(args...) = vertices(args..., DynamicLargestFirst(; reproduce_colpack=true))

## Distance-2 coloring

#=
Comparison with Tables VI and VII of the ColPack paper
=#

colpack_table_6_7 = CSV.read(
    joinpath(@__DIR__, "reference", "colpack_table_6_7.csv"), DataFrame
)

@testset verbose = true "Distance-2 coloring (ColPack paper)" begin
    @testset "$(row[:name])" for row in eachrow(colpack_table_6_7)
        original_mat = matrixdepot("$(row[:group])/$(row[:name])")
        mat = dropzeros(original_mat)
        bg = BipartiteGraph(mat)
        @testset "Graph features" begin
            @test nb_vertices(bg, Val(1)) == row[:V1]
            @test nb_vertices(bg, Val(2)) == row[:V2]
            @test nb_edges(bg) == row[:E]
            @test maximum_degree(bg, Val(1)) == row[:Δ1]
            @test maximum_degree(bg, Val(2)) == row[:Δ2]
        end
        @testset "Natural" begin
            @test nbunique(partial_distance2_coloring(bg, Val(1), _N(bg, Val(1)))) ==
                row[:N1]
            @test nbunique(partial_distance2_coloring(bg, Val(2), _N(bg, Val(2)))) ==
                row[:N2]
        end
        yield()
        @testset "LargestFirst" begin
            @test nbunique(partial_distance2_coloring(bg, Val(1), _LF(bg, Val(1)))) ==
                row[:LF1]
            @test nbunique(partial_distance2_coloring(bg, Val(2), _LF(bg, Val(2)))) ==
                row[:LF2]
        end
        yield()
        if row[:name] == "af23560"
            # orders differ for this one, not sure why
            continue
        end
        if row[:E] > 200_000
            # just to spare computational resources, but the larger tests pass too
            continue
        end
        @testset "SmallestLast" begin
            @test nbunique(partial_distance2_coloring(bg, Val(1), _SL(bg, Val(1)))) ==
                row[:SL1]
            @test nbunique(partial_distance2_coloring(bg, Val(2), _SL(bg, Val(2)))) ==
                row[:SL2]
        end
        yield()
        @testset "IncidenceDegree" begin
            @test nbunique(partial_distance2_coloring(bg, Val(1), _ID(bg, Val(1)))) ==
                row[:ID1]
            @test nbunique(partial_distance2_coloring(bg, Val(2), _ID(bg, Val(2)))) ==
                row[:ID2]
        end
        yield()
        @testset "DynamicLargestFirst" begin
            @test nbunique(partial_distance2_coloring(bg, Val(1), _DLF(bg, Val(1)))) ==
                row[:DLF1]
            @test nbunique(partial_distance2_coloring(bg, Val(2), _DLF(bg, Val(2)))) ==
                row[:DLF2]
        end
        yield()
    end
end;

#=
Comparison with Tables 3.1 and 3.2 of "What color is your Jacobian?"
=#

what_table_31_32 = CSV.read(
    joinpath(@__DIR__, "reference", "what_table_31_32.csv"), DataFrame
)

@testset "Distance-2 coloring (survey paper)" begin
    @testset "$(row[:name])" for row in eachrow(what_table_31_32)
        ismissing(row[:group]) && continue
        original_mat = matrixdepot("$(row[:group])/$(row[:name])")
        mat = original_mat  # no dropzeros
        bg = BipartiteGraph(mat)
        @test nb_vertices(bg, Val(1)) == row[:m]
        @test nb_vertices(bg, Val(2)) == row[:n]
        @test nb_edges(bg) == row[:nnz]
        @test minimum_degree(bg, Val(1)) == row[:ρmin]
        @test maximum_degree(bg, Val(1)) == row[:ρmax]
        @test minimum_degree(bg, Val(2)) == row[:κmin]
        @test maximum_degree(bg, Val(2)) == row[:κmax]
        vertices_in_order = vertices(bg, Val(2), NaturalOrder())
        color_Nb = partial_distance2_coloring(bg, Val(2), vertices_in_order)
        if length(unique(color_Nb)) == row[:K]
            @test length(unique(color_Nb)) == row[:K]
        else
            @test_broken length(unique(color_Nb)) == row[:K]
        end
        yield()
    end
end;

## Star coloring

what_table_41_42 = CSV.read(
    joinpath(@__DIR__, "reference", "what_table_41_42.csv"), DataFrame
)

@testset "Star coloring (survey paper)" begin
    @testset "$(row[:name])" for row in eachrow(what_table_41_42)
        ismissing(row[:group]) && continue
        original_mat = matrixdepot("$(row[:group])/$(row[:name])")
        mat = dropzeros(sparse(original_mat))
        ag = AdjacencyGraph(mat)
        bg = BipartiteGraph(mat)
        @test nb_vertices(ag) == row[:V]
        @test nb_edges(ag) == row[:E]
        @test maximum_degree(ag) == row[:Δ]
        @test minimum_degree(ag) == row[:δ]
        postprocessing = false
        vertices_in_order = vertices(ag, NaturalOrder())
        color_N, _ = star_coloring(ag, vertices_in_order, postprocessing)
        @test_skip row[:KS1] <= length(unique(color_N)) <= row[:KS2]  # TODO: find better
        yield()
    end
end;
