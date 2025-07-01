using ArrayInterface: ArrayInterface
using BandedMatrices: BandedMatrix
using Base: promote_eltype
using BlockBandedMatrices: BlockBandedMatrix
using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings:
    AdjacencyGraph,
    LinearSystemColoringResult,
    directly_recoverable_columns,
    matrix_versions,
    respectful_similar,
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    structurally_biorthogonal
using Test

function test_coloring_decompression(
    A0::AbstractMatrix,
    problem::ColoringProblem{structure,partition},
    algo::GreedyColoringAlgorithm{decompression};
    B0=nothing,
    color0=nothing,
    test_fast=false,
    gpu=false,
) where {structure,partition,decompression}
    color_vec = Vector{Int}[]
    @testset "$(typeof(A))" for A in matrix_versions(A0)
        yield()

        if structure == :nonsymmetric && issymmetric(A)
            result = coloring(
                A, problem, algo; decompression_eltype=Float32, symmetric_pattern=true
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

        @testset "Coherence" begin
            if partition == :column
                @test ncolors(result) == size(B, 2)
            elseif partition == :row
                @test ncolors(result) == size(B, 1)
            end
            if test_fast
                @test color == fast_coloring(A, problem, algo; symmetric_pattern=false)
            end
        end

        @testset "Reference" begin
            @test sparsity_pattern(result) === A  # identity of objects
            !isnothing(color0) && @test color == color0
            !isnothing(B0) && @test B == B0
        end

        @testset "Full decompression" begin
            @test decompress(B, result) ≈ A0
            @test decompress(B, result) ≈ A0  # check result wasn't modified
            @test decompress!(respectful_similar(A, eltype(B)), B, result) ≈ A0
            @test decompress!(respectful_similar(A, eltype(B)), B, result) ≈ A0
        end

        if gpu
            continue
        end

        @testset "Recoverability" begin
            # TODO: find tests for recoverability for substitution decompression
            if decompression == :direct
                if structure == :nonsymmetric
                    if partition == :column
                        @test structurally_orthogonal_columns(A0, color)
                        @test directly_recoverable_columns(A0, color)
                    elseif partition == :row
                        @test structurally_orthogonal_columns(transpose(A0), color)
                        @test directly_recoverable_columns(transpose(A0), color)
                    end
                else
                    # structure == :symmetric
                    if partition == :column
                        @test symmetrically_orthogonal_columns(A0, color)
                        @test directly_recoverable_columns(A0, color)
                    end
                end
            end
        end

        @testset "Single-color decompression" begin
            if decompression == :direct  # TODO: implement for :substitution too
                A2 = respectful_similar(A, eltype(B))
                A2 .= zero(eltype(A2))
                for c in unique(color)
                    c == 0 && continue
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
                A3upper = respectful_similar(triu(A), eltype(B))
                A3lower = respectful_similar(tril(A), eltype(B))
                A3both = respectful_similar(A, eltype(B))
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
                A4upper = respectful_similar(triu(A), eltype(B))
                A4lower = respectful_similar(tril(A), eltype(B))
                A4both = respectful_similar(A, eltype(B))
                A4upper .= zero(eltype(A))
                A4lower .= zero(eltype(A))
                A4both .= zero(eltype(A))

                for c in unique(color)
                    c == 0 && continue
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
                @test sparsity_pattern(result) === A  # identity of objects
                @test decompress(float.(B), linresult) ≈ A0
                @test decompress!(
                    respectful_similar(A, float(eltype(B))), float.(B), linresult
                ) ≈ A0
            end
        end
    end

    @testset "Coherence between all colorings" begin
        @test all(color_vec .== Ref(color_vec[1]))
        if !all(color_vec .== Ref(color_vec[1]))
            @show color_vec
        end
    end
end

function test_bicoloring_decompression(
    A0::AbstractMatrix,
    problem::ColoringProblem{:nonsymmetric,:bidirectional},
    algo::GreedyColoringAlgorithm{decompression};
    test_fast=false,
) where {decompression}
    @testset "$(typeof(A))" for A in matrix_versions(A0)
        yield()
        if issymmetric(A)
            result = coloring(
                A, problem, algo; decompression_eltype=Float32, symmetric_pattern=true
            )
        else
            result = coloring(A, problem, algo; decompression_eltype=Float64)
        end
        Br, Bc = compress(A, result)
        row_color, column_color = row_colors(result), column_colors(result)

        @testset "Coherence" begin
            @test size(Br, 1) == length(unique(row_color[row_color .> 0]))
            @test size(Bc, 2) == length(unique(column_color[column_color .> 0]))
            @test ncolors(result) == size(Br, 1) + size(Bc, 2)
            if test_fast
                @test (row_color, column_color) ==
                    fast_coloring(A, problem, algo; symmetric_pattern=false)
            end
        end

        @testset "Full decompression" begin
            @test decompress(Br, Bc, result) ≈ A0
            @test decompress(Br, Bc, result) ≈ A0  # check result wasn't modified
            @test decompress!(
                respectful_similar(A, promote_eltype(Br, Bc)), Br, Bc, result
            ) ≈ A0
            @test decompress!(
                respectful_similar(A, promote_eltype(Br, Bc)), Br, Bc, result
            ) ≈ A0
        end

        if decompression == :direct
            @testset "Recoverability" begin
                @test structurally_biorthogonal(A0, row_color, column_color)
            end
        end
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
    @test color == ArrayInterface.matrix_colors(A)

    # Row
    result = coloring(A, row_problem, algo)
    B = compress(A, result)
    D = decompress(B, result)
    @test D == A
    @test nameof(typeof(D)) == nameof(typeof(A))
end
