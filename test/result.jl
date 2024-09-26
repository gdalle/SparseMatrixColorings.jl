using SparseMatrixColorings: group_by_color
using Test

@testset "Group by color" begin
    for n in 10 .^ (2, 3, 4), cmax in (1, 2, 10, 100)
        color = rand(1:cmax, n)
        group = group_by_color(color)
        @test length(group) == maximum(color)
        @test all(1:maximum(color)) do c
            all(color[group[c]] .== c)
        end
    end
end
