using Aqua
using Documenter
using JET
using JuliaFormatter
using SparseMatrixColorings
using Test

@testset verbose = true "SparseMatrixColorings" begin
    if VERSION >= v"1.10"
        @testset verbose = true "Code quality" begin
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
    end
    @testset "Doctests" begin
        Documenter.doctest(SparseMatrixColorings)
    end
    @testset "Utils" begin
        include("utils.jl")
    end
    @testset "Check" begin
        include("check.jl")
    end
    @testset "ADTypes" begin
        include("adtypes.jl")
    end
end
