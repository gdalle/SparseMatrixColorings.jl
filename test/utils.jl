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
    for A in matrix_versions(A0)
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
        @test decompress(B, result) ≈ A0
        @test decompress(B, result) ≈ A0  # check result wasn't modified
        @test decompress!(respectful_similar(A), B, result) ≈ A0
        @test decompress!(respectful_similar(A), B, result) ≈ A0
        if decompression == :direct
            A2 = respectful_similar(A)
            A2 .= zero(eltype(A2))
            bs = if partition == :column
                eachcol(B)
            else
                eachrow(B)
            end
            for c in color
                decompress_single_color!(A2, bs[c], c, result)
            end
            @test A2 ≈ A0
        end
        if structure == :symmetric
            linresult = LinearSystemColoringResult(sparse(A), color, eltype(A))
            @test decompress(B, linresult) ≈ A0
            @test decompress!(respectful_similar(A), B, linresult) ≈ A0
        end
    end
    @test all(color_vec .== Ref(color_vec[1]))
end
