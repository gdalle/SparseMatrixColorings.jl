using SparseMatrixColorings: group_by_color, UnsupportedDecompressionError
using Test

@testset "Group by color" begin
    for n in 10 .^ (2, 3, 4), cmax in (1, 2, 10, 100), iteration in 1:10
        color = rand(1:cmax, n)
        group = group_by_color(color)
        @test length(group) == maximum(color)
        @test all(1:maximum(color)) do c
            all(color[group[c]] .== c) && issorted(group[c])
        end

        color = rand(0:cmax, n)
        group = group_by_color(color)
        @test length(group) == maximum(color)
        @test all(1:maximum(color)) do c
            all(color[group[c]] .== c) && issorted(group[c])
        end
    end
end

@testset "Empty compression" begin
    A = rand(10, 10)
    color = zeros(Int, 10)
    problem = ColoringProblem{:nonsymmetric,:column}()
    algo = ConstantColoringAlgorithm(A, color; partition=:column)
    B = compress(A, coloring(A, problem, algo))
    @test size(B, 2) == 0
    problem = ColoringProblem{:nonsymmetric,:row}()
    algo = ConstantColoringAlgorithm(A, color; partition=:row)
    B = compress(A, coloring(A, problem, algo))
    @test size(B, 1) == 0
end

@testset "Errors" begin
    e = SparseMatrixColorings.UnsupportedDecompressionError("hello")
    @test sprint(showerror, e) == "UnsupportedDecompressionError: hello"
end
