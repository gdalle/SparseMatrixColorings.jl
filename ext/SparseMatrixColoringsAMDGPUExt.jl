module SparseMatrixColoringsAMDGPUExt

import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using AMDGPU: ROCVector, ROCMatrix
using AMDGPU.rocSPARSE: AbstractROCSparseMatrix, ROCSparseMatrixCSC, ROCSparseMatrixCSR

SMC.matrix_versions(A::AbstractROCSparseMatrix) = (A,)

## Compression (slow, through CPU)

function SMC.compress(
    A::AbstractROCSparseMatrix, result::SMC.AbstractColoringResult{structure,:column}
) where {structure}
    return ROCMatrix(SMC.compress(SparseMatrixCSC(A), result))
end

function SMC.compress(
    A::AbstractROCSparseMatrix, result::SMC.AbstractColoringResult{structure,:row}
) where {structure}
    return ROCMatrix(SMC.compress(SparseMatrixCSC(A), result))
end

## CSC Result

function SMC.ColumnColoringResult(
    A::ROCSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=ROCVector(compressed_indices))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::ROCSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=ROCVector(compressed_indices))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::ROCSparseMatrixCSC,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csc=ROCVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## CSR Result

function SMC.ColumnColoringResult(
    A::ROCSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    compressed_indices_csr = SMC.column_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=ROCVector(compressed_indices_csr))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::ROCSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    compressed_indices_csr = SMC.row_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=ROCVector(compressed_indices_csr))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::ROCSparseMatrixCSR,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csr=ROCVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## Decompression

for R in (:ColumnColoringResult, :RowColoringResult)
    # loop to avoid method ambiguity
    @eval function SMC.decompress!(
        A::ROCSparseMatrixCSC, B::ROCMatrix, result::SMC.$R{<:ROCSparseMatrixCSC}
    )
        compressed_indices = result.additional_info.compressed_indices_gpu_csc
        copyto!(A.nzVal, view(B, compressed_indices))
        return A
    end

    @eval function SMC.decompress!(
        A::ROCSparseMatrixCSR, B::ROCMatrix, result::SMC.$R{<:ROCSparseMatrixCSR}
    )
        compressed_indices = result.additional_info.compressed_indices_gpu_csr
        copyto!(A.nzVal, view(B, compressed_indices))
        return A
    end
end

function SMC.decompress!(
    A::ROCSparseMatrixCSC,
    B::ROCMatrix,
    result::SMC.StarSetColoringResult{<:ROCSparseMatrixCSC},
    uplo::Symbol=:F,
)
    if uplo != :F
        throw(
            SMC.UnsupportedDecompressionError(
                "Single-triangle decompression is not supported on GPU matrices"
            ),
        )
    end
    compressed_indices = result.additional_info.compressed_indices_gpu_csc
    copyto!(A.nzVal, view(B, compressed_indices))
    return A
end

function SMC.decompress!(
    A::ROCSparseMatrixCSR,
    B::ROCMatrix,
    result::SMC.StarSetColoringResult{<:ROCSparseMatrixCSR},
    uplo::Symbol=:F,
)
    if uplo != :F
        throw(
            SMC.UnsupportedDecompressionError(
                "Single-triangle decompression is not supported on GPU matrices"
            ),
        )
    end
    compressed_indices = result.additional_info.compressed_indices_gpu_csr
    copyto!(A.nzVal, view(B, compressed_indices))
    return A
end

end
