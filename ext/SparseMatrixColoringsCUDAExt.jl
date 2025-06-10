module SparseMatrixColoringsCUDAExt

import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using CUDA:
    @cuda, CuVector, CuMatrix, blockIdx, blockDim, gridDim, threadIdx, launch_configuration
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

function update_nzval_from_matrix!(
    nzVal::AbstractVector, B::AbstractMatrix, compressed_indices::AbstractVector{<:Integer}
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for k in index:stride:length(nzVal)
        nzVal[k] = B[compressed_indices[k]]
    end
    return nothing
end

for R in (:ColumnColoringResult, :RowColoringResult, :StarSetColoringResult)
    # loop to avoid method ambiguity
    @eval function SMC.decompress!(
        A::CuSparseMatrixCSC, B::CuMatrix, result::SMC.$R{<:CuSparseMatrixCSC}
    )
        A.nnz == 0 && return A
        kernel = @cuda launch = false update_nzval_from_matrix!(
            A.nzVal, B, result.compressed_indices
        )
        config = launch_configuration(kernel.fun)
        threads = min(A.nnz, config.threads)
        blocks = cld(A.nnz, threads)
        kernel(A.nzVal, B, result.compressed_indices; threads, blocks)
        return A
    end
end

end
