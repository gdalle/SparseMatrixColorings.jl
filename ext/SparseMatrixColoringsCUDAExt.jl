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

## CSC Result

function SMC.ColumnColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::CuSparseMatrixCSC,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## CSR Result

function SMC.ColumnColoringResult(
    A::CuSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    compressed_indices_csr = SMC.column_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices_csr))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::CuSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    compressed_indices_csr = SMC.row_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices_csr))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::CuSparseMatrixCSR,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## Decompression

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
        compressed_indices = result.additional_info.compressed_indices_gpu_csc
        A.nnz == 0 && return A
        kernel = @cuda launch = false update_nzval_from_matrix!(
            A.nzVal, B, compressed_indices
        )
        config = launch_configuration(kernel.fun)
        threads = min(A.nnz, config.threads)
        blocks = cld(A.nnz, threads)
        kernel(A.nzVal, B, compressed_indices; threads, blocks)
        return A
    end

    @eval function SMC.decompress!(
        A::CuSparseMatrixCSR, B::CuMatrix, result::SMC.$R{<:CuSparseMatrixCSR}
    )
        compressed_indices = result.additional_info.compressed_indices_gpu_csr
        A.nnz == 0 && return A
        kernel = @cuda launch = false update_nzval_from_matrix!(
            A.nzVal, B, compressed_indices
        )
        config = launch_configuration(kernel.fun)
        threads = min(A.nnz, config.threads)
        blocks = cld(A.nnz, threads)
        kernel(A.nzVal, B, compressed_indices; threads, blocks)
        return A
    end
end

end
