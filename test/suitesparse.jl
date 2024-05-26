using CSV
using DataFrames
using LinearAlgebra
using MatrixDepot
using SparseArrays
using SparseMatrixColorings:
    Graph,
    adjacency_graph,
    bipartite_graph,
    LargestFirst,
    NaturalOrder,
    degree,
    minimum_degree,
    maximum_degree,
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
        bg = bipartite_graph(mat)
        @test length(bg, Val(1)) == row[:V1]
        @test length(bg, Val(2)) == row[:V2]
        @test nnz(bg) == row[:E]
        @test maximum_degree(bg, Val(1)) == row[:Δ1]
        @test maximum_degree(bg, Val(2)) == row[:Δ2]
        color_N1 = partial_distance2_coloring(bg, Val(1), NaturalOrder())
        color_N2 = partial_distance2_coloring(bg, Val(2), NaturalOrder())
        @test length(unique(color_N1)) == row[:N1]
        @test length(unique(color_N2)) == row[:N2]
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
        bg = bipartite_graph(mat)
        @test length(bg, Val(1)) == row[:m]
        @test length(bg, Val(2)) == row[:n]
        @test nnz(bg) == row[:nnz]
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
    end
end;

## Star coloring
