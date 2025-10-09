using Aqua
using Documenter
using JET
using JuliaFormatter
using SparseMatrixColorings
using Test

# Load package extensions to test them with JET
using Colors: Colors

include("utils.jl")

@testset verbose = true "SparseMatrixColorings" begin
    if get(ENV, "JULIA_SMC_TEST_GROUP", nothing) == "GPU"
        @testset "CUDA" begin
            using CUDA
            include("cuda.jl")
        end
    else
        @testset verbose = true "Code quality" begin
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
            @testset "Doctests" begin
                Documenter.doctest(SparseMatrixColorings)
            end
        end
        @testset verbose = true "Internals" begin
            @testset "Graph" begin
                include("graph.jl")
            end
            @testset "Forest" begin
                include("forest.jl")
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
            @testset "Optimal coloring" begin
                include("optimal.jl")
            end
            @testset "ADTypes coloring algorithms" begin
                include("adtypes.jl")
            end
            @testset "Visualization" begin
                include("show_colors.jl")
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
            @testset "Type stability" begin
                include("type_stability.jl")
            end
            @testset "Allocations" begin
                include("allocations.jl")
            end
        end
    end
end
