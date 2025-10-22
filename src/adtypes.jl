## From ADTypes to SMC

function coloring(
    A::AbstractMatrix,
    problem::ColoringProblem{structure,partition},
    algo::ADTypes.AbstractColoringAlgorithm;
    decompression_eltype::Type{R}=Float64,
    symmetric_pattern::Bool=false,
) where {structure,partition,R}
    symmetric_pattern = symmetric_pattern || A isa Union{Symmetric,Hermitian}
    if structure == :nonsymmetric
        if partition == :column
            forced_colors = ADTypes.column_coloring(A, algo)
        elseif partition == :row
            forced_colors = ADTypes.row_coloring(A, algo)
        else
            # TODO: improve once https://github.com/SciML/ADTypes.jl/issues/69 is done
            A_and_Aᵀ, _ = bidirectional_pattern(A; symmetric_pattern)
            forced_colors = ADTypes.symmetric_coloring(A_and_Aᵀ, algo)
        end
    else
        forced_colors = ADTypes.symmetric_coloring(A, algo)
    end
    return _coloring(
        WithResult(),
        A,
        problem,
        GreedyColoringAlgorithm(),
        R,
        symmetric_pattern;
        forced_colors,
    )
end
