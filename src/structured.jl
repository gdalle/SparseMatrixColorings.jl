#=
This code is partially taken from ArrayInterface.jl
https://github.com/JuliaArrays/ArrayInterface.jl
=#

"""
    cycle_range(k::Integer, n::Integer)

Concatenate copies of `1:k` to fill a vector of length `n` (with one partial copy allowed at the end).
"""
function cycle_range(k::Integer, n::Integer)
    color = Vector{Int}(undef, n)
    for i in eachindex(color)
        color[i] = 1 + (i - 1) % k
    end
    return color
end

## Diagonal

function coloring(
    A::Diagonal,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = fill(1, size(A, 2))
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function coloring(
    A::Diagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = fill(1, size(A, 1))
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

function decompress!(A::Diagonal, B::AbstractMatrix, result::ColumnColoringResult)
    color = column_colors(result)
    for j in axes(A, 2)
        A[j, j] = B[j, color[j]]
    end
    return A
end

function decompress!(A::Diagonal, B::AbstractMatrix, result::RowColoringResult)
    color = row_colors(result)
    for i in axes(A, 1)
        A[i, i] = B[color[i], i]
    end
    return A
end

## Bidiagonal

function coloring(
    A::Bidiagonal,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_range(2, size(A, 2))
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function coloring(
    A::Bidiagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_range(2, size(A, 1))
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

function decompress!(A::Bidiagonal, B::AbstractMatrix, result::ColumnColoringResult)
    color = column_colors(result)
    for j in axes(A, 2)
        c = color[j]
        A[j, j] = B[j, c]
        if A.uplo == 'U' && j > 1  # above
            A[j - 1, j] = B[j - 1, c]
        elseif A.uplo == 'L' && j < size(A, 2)  # below
            A[j + 1, j] = B[j + 1, c]
        end
    end
    return A
end

function decompress!(A::Bidiagonal, B::AbstractMatrix, result::RowColoringResult)
    color = row_colors(result)
    for i in axes(A, 1)
        c = color[i]
        A[i, i] = B[c, i]
        if A.uplo == 'U' && i < size(A, 1)  # right
            A[i, i + 1] = B[c, i + 1]
        elseif A.uplo == 'L' && i > 1  # left
            A[i, i - 1] = B[c, i - 1]
        end
    end
    return A
end

## Tridiagonal

function coloring(
    A::Tridiagonal,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_range(3, size(A, 2))
    bg = BipartiteGraph(A)
    return ColumnColoringResult(A, bg, color)
end

function coloring(
    A::Tridiagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_range(3, size(A, 1))
    bg = BipartiteGraph(A)
    return RowColoringResult(A, bg, color)
end

function decompress!(A::Tridiagonal, B::AbstractMatrix, result::ColumnColoringResult)
    color = column_colors(result)
    for j in axes(A, 2)
        c = color[j]
        A[j, j] = B[j, c]
        if j > 1  # above
            A[j - 1, j] = B[j - 1, c]
        end
        if j < size(A, 2)  # below
            A[j + 1, j] = B[j + 1, c]
        end
    end
    return A
end

function decompress!(A::Tridiagonal, B::AbstractMatrix, result::RowColoringResult)
    color = row_colors(result)
    for i in axes(A, 1)
        c = color[i]
        A[i, i] = B[c, i]
        if i < size(A, 1)  # right
            A[i, i + 1] = B[c, i + 1]
        end
        if i > 1  # left
            A[i, i - 1] = B[c, i - 1]
        end
    end
    return A
end
