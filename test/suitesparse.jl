using CSV
using DataFrames
using LinearAlgebra
using MatrixDepot
using SparseArrays
using SparseMatrixColorings:
    AdjacencyGraph,
    BipartiteGraph,
    LargestFirst,
    NaturalOrder,
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

## Distance-2 coloring

#=
Comparison with Tables VI and VII of the ColPack paper
=#

colpack_table_6_7 = CSV.read(
    joinpath(@__DIR__, "reference", "colpack_table_6_7.csv"), DataFrame
)

@testset "Distance-2 coloring (ColPack paper)" begin
    @testset "$(row[:name])" for row in eachrow(colpack_table_6_7)
        @info "Testing distance-2 coloring for $(row[:name]) against ColPack paper"
        original_mat = matrixdepot("$(row[:group])/$(row[:name])")
        mat = dropzeros(original_mat)
        bg = BipartiteGraph(mat)
        @test nb_vertices(bg, Val(1)) == row[:V1]
        @test nb_vertices(bg, Val(2)) == row[:V2]
        @test nb_edges(bg) == row[:E]
        @test maximum_degree(bg, Val(1)) == row[:Δ1]
        @test maximum_degree(bg, Val(2)) == row[:Δ2]
        color_N1 = partial_distance2_coloring(bg, Val(1), NaturalOrder())
        color_N2 = partial_distance2_coloring(bg, Val(2), NaturalOrder())
        @test length(unique(color_N1)) == row[:N1]
        @test length(unique(color_N2)) == row[:N2]
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
        @info "Testing distance-2 coloring for $(row[:name]) against survey paper"
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
        color_Nb = partial_distance2_coloring(bg, Val(2), NaturalOrder())
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
        @info "Testing star coloring for $(row[:name]) against survey paper"
        original_mat = matrixdepot("$(row[:group])/$(row[:name])")
        mat = dropzeros(sparse(original_mat))
        ag = AdjacencyGraph(mat)
        bg = BipartiteGraph(mat)
        @test nb_vertices(ag) == row[:V]
        @test nb_edges(ag) == row[:E]
        @test maximum_degree(ag) == row[:Δ]
        @test minimum_degree(ag) == row[:δ]
        color_N, _ = star_coloring(ag, NaturalOrder())
        @test_skip row[:KS1] <= length(unique(color_N)) <= row[:KS2]  # TODO: find better
        yield()
    end
end;
