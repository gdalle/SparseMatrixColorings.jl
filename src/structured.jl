#=
This code is partially taken from ArrayInterface.jl
https://github.com/JuliaArrays/ArrayInterface.jl

Question: when decompressing, should we always assume that the coloring was optimal?
=#

"""
    cycle_until(iterator, max_length::Integer)

Concatenate copies of `iterator` to fill a vector of length `max_length` (with one partial copy allowed at the end).
"""
function cycle_until(iterator, max_length::Integer)
    a = repeat(iterator, div(max_length, length(iterator)) + 1)
    return resize!(a, max_length)
end

## Diagonal

function coloring(
    A::Diagonal,
    ::ColoringProblem{:nonsymmetric,:column},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = fill(1, size(A, 2))
    return ColumnColoringResult(A, color)
end

function coloring(
    A::Diagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = fill(1, size(A, 1))
    return RowColoringResult(A, color)
end

function decompress!(
    A::Diagonal{R}, B::AbstractMatrix{R}, result::ColumnColoringResult
) where {R<:Real}
    color = column_colors(result)
    for j in axes(A, 2)
        A[j, j] = B[j, color[j]]
    end
    return A
end

function decompress!(
    A::Diagonal{R}, B::AbstractMatrix{R}, result::RowColoringResult
) where {R<:Real}
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
    color = cycle_until(1:2, size(A, 2))
    return ColumnColoringResult(A, color)
end

function coloring(
    A::Bidiagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_until(1:2, size(A, 1))
    return RowColoringResult(A, color)
end

function decompress!(
    A::Bidiagonal{R}, B::AbstractMatrix{R}, result::ColumnColoringResult
) where {R<:Real}
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

function decompress!(
    A::Bidiagonal{R}, B::AbstractMatrix{R}, result::RowColoringResult
) where {R<:Real}
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
    color = cycle_until(1:3, size(A, 2))
    return ColumnColoringResult(A, color)
end

function coloring(
    A::Tridiagonal,
    ::ColoringProblem{:nonsymmetric,:row},
    algo::GreedyColoringAlgorithm;
    kwargs...,
)
    color = cycle_until(1:3, size(A, 1))
    return RowColoringResult(A, color)
end

function decompress!(
    A::Tridiagonal{R}, B::AbstractMatrix{R}, result::ColumnColoringResult
) where {R<:Real}
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

function decompress!(
    A::Tridiagonal{R}, B::AbstractMatrix{R}, result::RowColoringResult
) where {R<:Real}
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
