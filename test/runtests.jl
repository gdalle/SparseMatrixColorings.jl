using Aqua
using Documenter
using JET
using JuliaFormatter
using SparseMatrixColorings
using Test

include("utils.jl")

@testset verbose = true "SparseMatrixColorings" begin
    @testset verbose = true "Code quality" begin
        if VERSION >= v"1.10"
            @testset "Aqua" begin
                Aqua.test_all(SparseMatrixColorings; stale_deps=(; ignore=[:Requires],))
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
        @testset "Matrices" begin
            include("matrices.jl")
        end
        @testset "Constructors" begin
            include("constructors.jl")
        end
        @testset "Result" begin
            include("result.jl")
        end
        @testset "Constant coloring" begin
            include("constant.jl")
        end
        @testset "ADTypes coloring algorithms" begin
            include("adtypes.jl")
        end
    end
    @testset verbose = true "Correctness" begin
        @testset "Small instances" begin
            include("small.jl")
        end
        @testset "Random instances" begin
            include("random.jl")
        end
        @testset "Structured matrices" begin
            include("structured.jl")
        end
        @testset "Instances with known colorings" begin
            include("theory.jl")
        end
        @testset "SuiteSparse" begin
            include("suitesparse.jl")
        end
    end
    @testset verbose = true "Performance" begin
        if VERSION >= v"1.10"
            @testset "Type stability" begin
                include("type_stability.jl")
            end
        end
        @testset "Allocations" begin
            include("allocations.jl")
        end
    end
end
