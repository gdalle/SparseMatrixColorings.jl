using SparseMatrixColorings:
    check_structurally_orthogonal_columns, check_symmetrically_orthogonal_columns
using Test

@testset "Structurally orthogonal columns" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]

    # success

    @test check_structurally_orthogonal_columns(A, [1, 2, 3])
    @test check_structurally_orthogonal_columns(A, [1, 2, 1])
    @test check_structurally_orthogonal_columns(A, [1, 1, 2])

    # failure

    @test !check_structurally_orthogonal_columns(A, [1, 2])
    @test_logs (:warn, "2 colors provided for 3 columns") check_structurally_orthogonal_columns(
        A, [1, 2]; verbose=true
    )

    @test !check_structurally_orthogonal_columns(A, [1, 2, 2])
    @test_logs (:warn, "In color 2, columns [2, 3] all have nonzeros in row 3") check_structurally_orthogonal_columns(
        A, [1, 2, 2]; verbose=true
    )
end

@testset "Structurally orthogonal rows" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]

    # success

    @test check_structurally_orthogonal_columns(transpose(A), [1, 2, 3])
    @test check_structurally_orthogonal_columns(transpose(A), [1, 2, 1])
    @test check_structurally_orthogonal_columns(transpose(A), [1, 1, 2])

    # failure

    @test !check_structurally_orthogonal_columns(transpose(A), [1, 2, 2, 3])
    @test_logs (:warn, "4 colors provided for 3 columns") check_structurally_orthogonal_columns(
        transpose(A), [1, 2, 2, 3]; verbose=true
    )

    @test !check_structurally_orthogonal_columns(transpose(A), [1, 2, 2])
    @test_logs (:warn, "In color 2, columns [2, 3] all have nonzeros in row 2") !check_structurally_orthogonal_columns(
        transpose(A), [1, 2, 2]; verbose=true
    )
end

@testset "Symmetrically orthogonal" begin
    # Fig 4.1 of "What color is your Jacobian?"

    A = [
        1 1 0 0 0 0
        1 1 1 0 1 1
        0 1 1 1 0 0
        0 0 1 1 0 1
        0 1 0 0 1 0
        0 1 0 1 0 1
    ]
    @test issymmetric(A)

    # success

    @test check_symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 1])

    # failure

    @test !check_symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1])
    @test_logs (:warn, "5 colors provided for 6 columns") check_symmetrically_orthogonal_columns(
        A, [1, 2, 1, 3, 1]; verbose=true
    )

    @test !check_symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1])
    @test_logs (
        :warn,
        """
For coefficient (i=2, j=3) with column colors (ci=3, cj=1):
- in color ci=3, columns [2, 4] all have nonzeros in row j=3
- in color cj=1, columns [1, 3, 5, 6] all have nonzeros in row i=2
""",
    ) check_symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1]; verbose=true)

    # Fig 1 of "Efficient computation of sparse hessians using coloring and AD"

    A = [
        1 1 0 0 0 0 1 0 0 0
        1 1 1 0 1 0 0 0 0 0
        0 1 1 1 0 1 0 0 0 0
        0 0 1 1 0 0 0 0 0 1
        0 1 0 0 1 1 0 1 0 0
        0 0 1 0 1 1 0 0 1 0
        1 0 0 0 0 0 1 1 0 0
        0 0 0 0 1 0 1 1 1 0
        0 0 0 0 0 1 0 1 1 1
        0 0 0 1 0 0 0 0 1 1
    ]
    @test issymmetric(A)

    # success

    @test check_symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 5, 1, 2])

    # failure

    @test !check_symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 4, 1, 2])
    @test !check_symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 2, 5, 1, 2])
    @test !check_symmetrically_orthogonal_columns(A, [1, 2, 1, 4, 1, 4, 3, 5, 1, 2])
end
