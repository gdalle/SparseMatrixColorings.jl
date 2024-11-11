using LinearAlgebra
using SparseMatrixColorings:
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    structurally_biorthogonal,
    directly_recoverable_columns,
    what_fig_41,
    efficient_fig_1
using Test

@testset "Structurally orthogonal columns" begin
    A = [
        1 0 0
        0 2 0
        0 3 4
    ]

    # success

    @test structurally_orthogonal_columns(A, [1, 2, 3])
    @test structurally_orthogonal_columns(A, [1, 2, 1])
    @test structurally_orthogonal_columns(A, [1, 1, 2])

    @test directly_recoverable_columns(A, [1, 2, 3])
    @test directly_recoverable_columns(A, [1, 2, 1])
    @test directly_recoverable_columns(A, [1, 1, 2])

    # failure

    @test !structurally_orthogonal_columns(A, [1, 2])
    log = (:warn, "2 colors provided for 3 columns.")
    @test_logs log structurally_orthogonal_columns(A, [1, 2]; verbose=true)

    @test !directly_recoverable_columns(A, [1, 2])
    log = (:warn, "2 colors provided for 3 columns.")
    @test_logs log !directly_recoverable_columns(A, [1, 2]; verbose=true)

    @test !structurally_orthogonal_columns(A, [1, 2, 2])
    log = (:warn, "In color 2, columns [2, 3] all have nonzeros in row 3.")
    @test_logs log structurally_orthogonal_columns(A, [1, 2, 2]; verbose=true)

    @test !directly_recoverable_columns(A, [1, 2, 2])
    log = (:warn, "Coefficients [3, 4] are not directly recoverable.")
end

@testset "Structurally orthogonal rows" begin
    A = [
        1 0 0
        0 2 0
        0 3 4
    ]

    # success

    @test structurally_orthogonal_columns(transpose(A), [1, 2, 3])
    @test structurally_orthogonal_columns(transpose(A), [1, 2, 1])
    @test structurally_orthogonal_columns(transpose(A), [1, 1, 2])

    @test directly_recoverable_columns(transpose(A), [1, 2, 3])
    @test directly_recoverable_columns(transpose(A), [1, 2, 1])
    @test directly_recoverable_columns(transpose(A), [1, 1, 2])

    # failure

    @test !structurally_orthogonal_columns(transpose(A), [1, 2, 2, 3])
    log = (:warn, "4 colors provided for 3 columns.")
    @test_logs log structurally_orthogonal_columns(transpose(A), [1, 2, 2, 3]; verbose=true)

    @test !directly_recoverable_columns(transpose(A), [1, 2, 2, 3])
    log = (:warn, "4 colors provided for 3 columns.")
    @test_logs log directly_recoverable_columns(transpose(A), [1, 2, 2, 3]; verbose=true)

    @test !structurally_orthogonal_columns(transpose(A), [1, 2, 2])
    log = (:warn, "In color 2, columns [2, 3] all have nonzeros in row 2.")
    @test_logs log !structurally_orthogonal_columns(transpose(A), [1, 2, 2]; verbose=true)

    @test !directly_recoverable_columns(transpose(A), [1, 2, 2])
    log = (:warn, "Coefficients [2, 3] are not directly recoverable.")
    @test_logs log directly_recoverable_columns(transpose(A), [1, 2, 2]; verbose=true)
end

@testset "Symmetrically orthogonal" begin
    A = what_fig_41().A
    @test issymmetric(A)

    # success

    @test symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 1])

    @test directly_recoverable_columns(A, [1, 2, 1, 3, 1, 1])

    # failure

    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1])
    @test_logs (:warn, "5 colors provided for 6 columns.") symmetrically_orthogonal_columns(
        A, [1, 2, 1, 3, 1]; verbose=true
    )

    @test !symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1])
    @test_logs (
        :warn,
        """
For coefficient (i=2, j=3) with column colors (ci=3, cj=1):
- In color ci=3, columns [2, 4] all have nonzeros in row j=3.
- In color cj=1, columns [1, 3, 5, 6] all have nonzeros in row i=2.
""",
    ) symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1]; verbose=true)

    A = efficient_fig_1().A
    @test issymmetric(A)

    # success

    @test symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 5, 1, 2])

    # failure

    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 4, 1, 2])
    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 2, 5, 1, 2])
    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 4, 1, 4, 3, 5, 1, 2])

    @test !directly_recoverable_columns(A, [1, 2, 1, 3, 1, 4, 3, 4, 1, 2])
    @test !directly_recoverable_columns(A, [1, 2, 1, 3, 1, 4, 2, 5, 1, 2])
    @test !directly_recoverable_columns(A, [1, 2, 1, 4, 1, 4, 3, 5, 1, 2])
end

@testset "Structurally biorthogonal" begin
    A = [
        1 5 7 9 11
        2 0 0 0 12
        3 0 0 0 13
        4 6 8 10 14
    ]

    # success

    @test structurally_biorthogonal(A, [1, 2, 2, 3], [1, 2, 2, 2, 3])

    # failure

    @test !structurally_biorthogonal(A, [1, 2, 2, 2, 3], [1, 2, 2, 3])
    @test !structurally_biorthogonal(A, [1, 1, 1, 2], [1, 1, 1, 1, 2])
end
