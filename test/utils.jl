using LinearAlgebra
using SparseMatrixColorings
using SparseMatrixColorings: LinearSystemColoringResult, matrix_versions, respectful_similar
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
        result = coloring(A, problem, algo; decompression_eltype=eltype(A))
        color = if partition == :column
            column_colors(result)
        elseif partition == :row
            row_colors(result)
        end
        push!(color_vec, color)

        B = compress(A, result)
        !isnothing(color0) && @test color == color0
        !isnothing(B0) && @test B == B0

        # Full decompression
        @test decompress(B, result) ≈ A0
        @test decompress(B, result) ≈ A0  # check result wasn't modified
        @test decompress!(respectful_similar(A), B, result) ≈ A0
        @test decompress!(respectful_similar(A), B, result) ≈ A0

        # Colorwise decompression
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

        # Triangle decompression
        if structure == :symmetric
            A3upper = respectful_similar(A)
            A3lower = respectful_similar(A)
            A3both = respectful_similar(A)
            A3upper .= zero(eltype(A))
            A3lower .= zero(eltype(A))
            A3both .= zero(eltype(A))

            decompress!(A3upper, B, result, :U)
            decompress!(A3lower, B, result, :L)
            decompress!(A3both, B, result, :UL)

            @test A3upper ≈ triu(A0)
            @test A3lower ≈ tril(A0)
            @test A3both ≈ A0
        end

        # Linear system decompression
        if structure == :symmetric
            linresult = LinearSystemColoringResult(sparse(A), color, eltype(A))
            @test decompress(B, linresult) ≈ A0
            @test decompress!(respectful_similar(A), B, linresult) ≈ A0
        end
    end

    # Coherence between all colorings
    @test all(color_vec .== Ref(color_vec[1]))
end
