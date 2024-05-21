using Aqua
using Documenter
using JET
using JuliaFormatter
using SparseMatrixColorings
using Test

@testset verbose = true "SparseMatrixColorings" begin
    @testset verbose = true "Code quality" begin
        if VERSION >= v"1.10"
            @testset "Aqua" begin
                Aqua.test_all(SparseMatrixColorings)
            end
            @testset "JET" begin
                JET.test_package(SparseMatrixColorings; target_defined_modules=true)
            end
            @testset "JuliaFormatter" begin
                @test JuliaFormatter.format(
                    SparseMatrixColorings; verbose=false, overwrite=false
                )
            end
        end
        @testset "Doctests" begin
            Documenter.doctest(SparseMatrixColorings)
        end
    end
    @testset verbose = true "Internals" begin
        @testset "Graph" begin
            include("graph.jl")
        end
        @testset "Order" begin
            include("order.jl")
        end
        @testset "Check" begin
            include("check.jl")
        end
    end
    @testset verbose = true "Correctness" begin
        @testset "ADTypes" begin
            include("adtypes.jl")
        end
        @testset "SuiteSparse" begin
            include("suitesparse.jl")
        end
    end
    @testset "Performance" begin
        if VERSION >= v"1.10"
            include("performance.jl")
        end
    end
end
