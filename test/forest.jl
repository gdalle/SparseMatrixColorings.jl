@testset "Constructor Forest" begin
    forest = Forest{Int}(5)

    @test forest.num_edges == 0
    @test forest.num_trees == 0
    @test length(forest.intmap) == 0
    @test length(forest.parents) == 5
    @test all(forest.parents .== 1:5)
    @test all(forest.ranks .== 0)
end

@testset "Push edge" begin
    forest = Forest{Int}(5)

    push!(forest, (1, 2))
    @test forest.num_edges == 1
    @test forest.num_trees == 1
    @test haskey(forest.intmap, (1, 2))
    @test forest.intmap[(1, 2)] == 1
    @test forest.num_trees == 1

    push!(forest, (3, 4))
    @test forest.num_edges == 2
    @test forest.num_trees == 2
    @test haskey(forest.intmap, (3, 4))
    @test forest.intmap[(3, 4)] == 2
    @test forest.num_trees == 2
end

@testset "Find root" begin
    forest = Forest{Int}(5)
    push!(forest, (1, 2))
    push!(forest, (3, 4))

    @test find_root!(forest, (1, 2)) == 1
    @test find_root!(forest, (3, 4)) == 2
end

@testset "Root union" begin
    forest = Forest{Int}(5)
    push!(forest, (1, 2))
    push!(forest, (4, 5))
    push!(forest, (2, 4))
    @test forest.num_trees = 3

    root1 = find_root!(forest, (1, 2))
    root3 = find_root!(forest, (2, 4))
    @test root1 != root3

    root_union!(forest, root1, root3)
    @test find_root!(forest, (2, 4)) == 1
    @test forest.parents[1] == 1
    @test forest.parents[3] == 1
    @test forest.ranks[1] == 1
    @test forest.ranks[3] == 0
    @test forest.num_trees = 2

    root1 = find_root!(forest, (1, 2))
    root2 = find_root!(forest, (4, 5))
    @test root1 != root2
    root_union!(forest, root1, root2)
    @test find_root!(forest, (4, 5)) == 1
    @test forest.parents[1] == 1
    @test forest.parents[2] == 1
    @test forest.ranks[1] == 1
    @test forest.ranks[2] == 0
    @test forest.num_trees == 1

    push!(forest, (1, 4))
    @test forest.num_trees == 2
    @test forest.intmap[(1, 4)] == 4
    @test forest.parents[4] == 4
end
