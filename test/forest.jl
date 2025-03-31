using SparseMatrixColorings: Forest, find_root!, root_union!
using Test

@testset "Constructor Forest" begin
    forest = Forest{Int}(5)
    @test forest.num_trees == 5
    @test length(forest.parents) == 5
    @test all(forest.parents .== 1:5)
    @test all(forest.ranks .== 0)
end

@testset "Find root" begin
    forest = Forest{Int}(5)
    @test find_root!(forest, 3) == 3
    @test find_root!(forest, 5) == 5
end

@testset "Root union" begin
    forest = Forest{Int}(5)
    root1 = find_root!(forest, 1)
    root3 = find_root!(forest, 3)
    @test root1 != root3

    root_union!(forest, root1, root3)
    @test find_root!(forest, 3) == 1
    @test forest.parents[1] == 1
    @test forest.parents[3] == 1
    @test forest.ranks[1] == 1
    @test forest.ranks[3] == 0
    @test forest.num_trees == 4

    root1 = find_root!(forest, 1)
    root2 = find_root!(forest, 2)
    @test root1 != root2

    root_union!(forest, root1, root2)
    @test find_root!(forest, 2) == 1
    @test forest.parents[1] == 1
    @test forest.parents[2] == 1
    @test forest.ranks[1] == 1
    @test forest.ranks[2] == 0
    @test forest.num_trees == 3
end
