module SparseMatrixColoringsCUDAExt

import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using CUDA: CuVector, CuMatrix
using CUDA.CUSPARSE: AbstractCuSparseMatrix, CuSparseMatrixCSC, CuSparseMatrixCSR

SMC.matrix_versions(A::AbstractCuSparseMatrix) = (A,)

## Compression (slow, through CPU)

function SMC.compress(
    A::AbstractCuSparseMatrix, result::SMC.AbstractColoringResult{structure,:column}
) where {structure}
    return CuMatrix(SMC.compress(SparseMatrixCSC(A), result))
end

function SMC.compress(
    A::AbstractCuSparseMatrix, result::SMC.AbstractColoringResult{structure,:row}
) where {structure}
    return CuMatrix(SMC.compress(SparseMatrixCSC(A), result))
end

## CSC

function SMC.ColumnColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    A_cpu = SparseMatrixCSC(A)
    result_cpu = SMC.ColumnColoringResult(A_cpu, bg, color)
    compressed_indices = CuVector(result_cpu.compressed_indices)
    return SMC.ColumnColoringResult(A, bg, color, result_cpu.group, compressed_indices)
end

function SMC.RowColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    A_cpu = SparseMatrixCSC(A)
    result_cpu = SMC.RowColoringResult(A_cpu, bg, color)
    compressed_indices = CuVector(result_cpu.compressed_indices)
    return SMC.RowColoringResult(A, bg, color, result_cpu.group, compressed_indices)
end

function SMC.StarSetColoringResult(
    A::CuSparseMatrixCSC,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    A_cpu = SparseMatrixCSC(A)
    result_cpu = SMC.StarSetColoringResult(A_cpu, ag, color, star_set)
    compressed_indices = CuVector(result_cpu.compressed_indices)
    return SMC.StarSetColoringResult(A, ag, color, result_cpu.group, compressed_indices)
end

# TODO: write a kernel

function SMC.decompress!(
    A::CuSparseMatrixCSC, B::CuMatrix, result::SMC.ColumnColoringResult{<:CuSparseMatrixCSC}
)
    A.nzVal .= getindex.(Ref(B), result.compressed_indices)
    return A
end

function SMC.decompress!(
    A::CuSparseMatrixCSC, B::CuMatrix, result::SMC.RowColoringResult{<:CuSparseMatrixCSC}
)
    A.nzVal .= getindex.(Ref(B), result.compressed_indices)
    return A
end

function SMC.decompress!(
    A::CuSparseMatrixCSC,
    B::CuMatrix,
    result::SMC.StarSetColoringResult{<:CuSparseMatrixCSC},
)
    A.nzVal .= getindex.(Ref(B), result.compressed_indices)
    return A
end

end
