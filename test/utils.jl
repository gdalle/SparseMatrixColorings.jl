using SparseMatrixColorings
using SparseMatrixColorings: ColoringProblem, DefaultColoringResult
using Test

function test_coloring_decompression(
    A0::AbstractMatrix,
    problem::ColoringProblem{structure,partition,decompression},
    algo::GreedyColoringAlgorithm;
    B0=nothing,
    color0=nothing,
) where {structure,partition,decompression}
    color_vec = Vector{Int}[]
    @testset "A::$(typeof(A))" for A in matrix_versions(A0)
        result = coloring(A, problem, algo)
        default_result = DefaultColoringResult(result)
        color = if partition == :column
            column_colors(result)
        elseif partition == :row
            row_colors(result)
        end
        push!(color_vec, color)
        B = compress(A, result)
        !isnothing(color0) && @test color == color0
        !isnothing(B0) && @test B == B0
        @test decompress(B, default_result) ≈ A0
        @test decompress(B, result) ≈ A0
        @test decompress(B, result) ≈ A0  # check result wasn't modified
        @test decompress!(respectful_similar(A), B, default_result) ≈ A0
        @test decompress!(respectful_similar(A), B, result) ≈ A0
        @test decompress!(respectful_similar(A), B, result) ≈ A0
    end
    @test all(color_vec .== Ref(color_vec[1]))
end
