using ArrayInterface: ArrayInterface
using BandedMatrices: BandedMatrix
using BlockBandedMatrices: BlockBandedMatrix
using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings:
    AdjacencyGraph, LinearSystemColoringResult, matrix_versions, respectful_similar
using Test

function test_coloring_decompression(
    A0::AbstractMatrix,
    problem::ColoringProblem{structure,partition},
    algo::GreedyColoringAlgorithm{decompression};
    B0=nothing,
    color0=nothing,
) where {structure,partition,decompression}
    color_vec = Vector{Int}[]
    @testset "$(typeof(A))" for A in matrix_versions(A0)
        yield()

        if structure == :nonsymmetric && issymmetric(A)
            result = coloring(
                A, problem, algo; decompression_eltype=Float64, symmetric_pattern=true
            )
        else
            result = coloring(A, problem, algo; decompression_eltype=Float64)
        end
        color = if partition == :column
            column_colors(result)
        elseif partition == :row
            row_colors(result)
        end
        push!(color_vec, color)

        B = compress(A, result)

        @testset "Reference" begin
            !isnothing(color0) && @test color == color0
            !isnothing(B0) && @test B == B0
        end

        @testset "Full decompression" begin
            @test decompress(B, result) ≈ A0
            @test decompress(B, result) ≈ A0  # check result wasn't modified
            @test decompress!(respectful_similar(A), B, result) ≈ A0
            @test decompress!(respectful_similar(A), B, result) ≈ A0
        end

        @testset "Single-color decompression" begin
            if decompression == :direct  # TODO: implement for :substitution too
                A2 = respectful_similar(A)
                A2 .= zero(eltype(A2))
                for c in unique(color)
                    if partition == :column
                        decompress_single_color!(A2, B[:, c], c, result)
                    elseif partition == :row
                        decompress_single_color!(A2, B[c, :], c, result)
                    end
                end
                @test A2 ≈ A0
            end
        end

        @testset "Triangle decompression" begin
            if structure == :symmetric
                A3upper = respectful_similar(triu(A))
                A3lower = respectful_similar(tril(A))
                A3both = respectful_similar(A)
                A3upper .= zero(eltype(A))
                A3lower .= zero(eltype(A))
                A3both .= zero(eltype(A))

                decompress!(A3upper, B, result, :U)
                decompress!(A3lower, B, result, :L)
                decompress!(A3both, B, result, :F)

                @test A3upper ≈ triu(A0)
                @test A3lower ≈ tril(A0)
                @test A3both ≈ A0
            end
        end

        @testset "Single-color triangle decompression" begin
            if structure == :symmetric && decompression == :direct
                A4upper = respectful_similar(triu(A))
                A4lower = respectful_similar(tril(A))
                A4both = respectful_similar(A)
                A4upper .= zero(eltype(A))
                A4lower .= zero(eltype(A))
                A4both .= zero(eltype(A))

                for c in unique(color)
                    decompress_single_color!(A4upper, B[:, c], c, result, :U)
                    decompress_single_color!(A4lower, B[:, c], c, result, :L)
                    decompress_single_color!(A4both, B[:, c], c, result, :F)
                end

                @test A4upper ≈ triu(A0)
                @test A4lower ≈ tril(A0)
                @test A4both ≈ A0
            end
        end

        @testset "Linear system decompression" begin
            if structure == :symmetric && count(!iszero, A) > 0  # sparse factorization cannot handle empty matrices
                ag = AdjacencyGraph(A)
                linresult = LinearSystemColoringResult(A, ag, color, Float64)
                @test decompress(float.(B), linresult) ≈ A0
                @test decompress!(respectful_similar(float.(A)), float.(B), linresult) ≈ A0
            end
        end
    end

    @testset "Coherence between all colorings" begin
        @test all(color_vec .== Ref(color_vec[1]))
    end
end

function test_structured_coloring_decompression(A::AbstractMatrix)
    column_problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    row_problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm()

    # Column
    result = coloring(A, column_problem, algo)
    color = column_colors(result)
    B = compress(A, result)
    D = decompress(B, result)
    @test D == A
    @test nameof(typeof(D)) == nameof(typeof(A))
    @test structurally_orthogonal_columns(A, color)
    if VERSION >= v"1.10" || A isa Union{Diagonal,Bidiagonal,Tridiagonal}
        # banded matrices not supported by ArrayInterface on Julia 1.6
        @test color == ArrayInterface.matrix_colors(A)
    end

    # Row
    result = coloring(A, row_problem, algo)
    B = compress(A, result)
    D = decompress(B, result)
    @test D == A
    @test nameof(typeof(D)) == nameof(typeof(A))
end
