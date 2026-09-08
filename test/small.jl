using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings:
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    directly_recoverable_columns,
    what_fig_41,
    what_fig_61,
    efficient_fig_1,
    efficient_fig_4
using StableRNGs
using Test

@testset "Column coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    A0 = sparse([
        1 0 2
        0 3 4
        5 0 0
    ])
    B0 = [
        1 2
        3 4
        5 0
    ]
    color0 = [1, 1, 2]
    @test structurally_orthogonal_columns(A0, color0)
    @test directly_recoverable_columns(A0, color0)
    test_coloring_decompression(A0, problem, algo; B0, color0, test_fast=true)
end;

@testset "Row coloring & decompression" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    algo = GreedyColoringAlgorithm(; decompression=:direct)
    A0 = sparse([
        1 0 3
        0 2 0
        4 5 0
    ])
    B0 = [
        1 2 3
        4 5 0
    ]
    color0 = [1, 1, 2]
    @test structurally_orthogonal_columns(transpose(A0), color0)
    @test directly_recoverable_columns(transpose(A0), color0)
    test_coloring_decompression(A0, problem, algo; B0, color0, test_fast=true)
end;

@testset "Symmetric coloring & direct decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:direct)

    @testset "Fig 4.1 from 'What color is your Jacobian'" begin
        example = what_fig_41()
        A0, B0, color0 = example.A, example.B, example.color
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; B0, color0, test_fast=true)
    end

    @testset "Fig 1 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_1()
        A0, B0, color0 = example.A, example.B, example.color
        @test symmetrically_orthogonal_columns(A0, color0)
        @test directly_recoverable_columns(A0, color0)
        test_coloring_decompression(A0, problem, algo; B0, color0, test_fast=true)
    end
end;

@testset "Symmetric coloring & substitution decompression" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algo = GreedyColoringAlgorithm(; decompression=:substitution)

    @testset "Fig 6.1 from 'What color is your Jacobian'" begin
        example = what_fig_61()
        A0, B0, color0 = example.A, example.B, example.color
        # our coloring doesn't give the color0 from the example, but that's okay
        test_coloring_decompression(A0, problem, algo; test_fast=true)
    end

    @testset "Fig 4 from 'Efficient computation of sparse hessians using coloring and AD'" begin
        example = efficient_fig_4()
        A0, B0, color0 = example.A, example.B, example.color
        test_coloring_decompression(A0, problem, algo; B0, color0, test_fast=true)
    end
end;

@testset "Bidirectional coloring" begin
    color_stats = Dict(
        "Anti-diagonal" => Dict(
            :direct => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
            :substitution => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
        ),
        "Triangle" => Dict(
            :direct => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
            :substitution => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
        ),
        "Rectangle" => Dict(
            :direct => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
            :substitution => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
        ),
        "Arrowhead" => Dict(
            :direct => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
            :substitution => Dict(
                :all_colors => (0, 0),
                :row_colors => (0, 0),
                :column_colors => (0, 0),
            ),
        ),
    )

    @testset "coverage postprocessing for acyclic bicoloring" begin
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
        order = RandomOrder(StableRNG(4), 4)
        algo = GreedyColoringAlgorithm(
            order; decompression=:substitution, postprocessing=true
        )
        A = sparse([1 1 0; 0 0 1; 0 1 0])
        result = coloring(A, problem, algo)
    end

    @testset "postprocessing_minimizes = :sym_colors" begin
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
        order = NaturalOrder()
        A = sparse([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])

        star_bicoloring_algorithm = GreedyColoringAlgorithm{:direct}(;
            postprocessing=true, postprocessing_minimizes=:sym_colors
        )
        @test_throws "The value postprocessing_minimizes = :sym_colors is not supported." coloring(
            A, problem, star_bicoloring_algorithm
        )

        acyclic_bicoloring_algorithm = GreedyColoringAlgorithm{:substitution}(;
            postprocessing=true, postprocessing_minimizes=:sym_colors
        )
        @test_throws "The value postprocessing_minimizes = :sym_colors is not supported." coloring(
            A, problem, acyclic_bicoloring_algorithm
        )
    end

    @testset "postprocessing_minimizes = $target" for target in (
        :all_colors, :row_colors, :column_colors
    )
        problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
        order = RandomOrder(StableRNG(0), 0)

        @testset "Anti-diagonal" begin
            A = sparse([0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0])

            result = coloring(
                A, problem, GreedyColoringAlgorithm{:direct}(; postprocessing=false)
            )
            @test ncolors(result) == 2

            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:direct}(;
                    postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 1
            color_stats["Anti-diagonal"][:direct][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )

            result = coloring(
                A, problem, GreedyColoringAlgorithm{:substitution}(; postprocessing=false)
            )
            @test ncolors(result) == 2

            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(;
                    postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 1
            color_stats["Anti-diagonal"][:substitution][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )
        end

        @testset "Triangle" begin
            A = sparse([1 1 0; 0 1 1; 1 0 1])

            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:direct}(;
                    postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 3
            color_stats["Triangle"][:direct][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )

            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(;
                    postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 3
            color_stats["Triangle"][:substitution][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )
        end

        @testset "Rectangle" begin
            A = spzeros(Bool, 10, 20)
            A[:, 1] .= 1
            A[:, end] .= 1
            A[1, :] .= 1
            A[end, :] .= 1

            result = coloring(
                A, problem, GreedyColoringAlgorithm{:direct}(order; postprocessing=false)
            )
            @test ncolors(result) == 6  # two more than necessary
            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:direct}(
                    order; postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 4  # optimal number
            color_stats["Rectangle"][:direct][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )

            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(order; postprocessing=false),
            )
            @test ncolors(result) == 6  # two more than necessary
            result = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(
                    order; postprocessing=true, postprocessing_minimizes=target
                ),
            )
            @test ncolors(result) == 4  # optimal number
            color_stats["Rectangle"][:substitution][target] = (
                maximum(row_colors(result)), maximum(column_colors(result))
            )
        end

        @testset "Arrowhead" begin
            A = spzeros(Bool, 10, 10)
            for i in axes(A, 1)
                A[1, i] = 1
                A[i, 1] = 1
                A[i, i] = 1
            end

            @test ncolors(
                coloring(A, problem, GreedyColoringAlgorithm{:substitution}(order))
            ) < ncolors(coloring(A, problem, GreedyColoringAlgorithm{:direct}(order)))

            result_direct = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:direct}(
                    order; postprocessing=true, postprocessing_minimizes=target
                ),
            )
            color_stats["Arrowhead"][:direct][target] = (
                maximum(row_colors(result_direct)), maximum(column_colors(result_direct))
            )

            result_substitution = coloring(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(
                    order; postprocessing=true, postprocessing_minimizes=target
                ),
            )
            color_stats["Arrowhead"][:substitution][target] = (
                maximum(row_colors(result_substitution)),
                maximum(column_colors(result_substitution)),
            )

            @test ncolors(result_substitution) < ncolors(result_direct)

            test_bicoloring_decompression(
                A,
                problem,
                GreedyColoringAlgorithm{:direct}(
                    order; postprocessing=true, postprocessing_minimizes=target
                );
                test_fast=true,
            )

            test_bicoloring_decompression(
                A,
                problem,
                GreedyColoringAlgorithm{:substitution}(
                    order; postprocessing=true, postprocessing_minimizes=target
                );
                test_fast=true,
            )
        end
    end

    @testset "Variants of post-processing" begin
        for problem in ("Anti-diagonal", "Triangle", "Rectangle", "Arrowhead")
            for mode in (:direct, :substitution)
                (num_row_colors1, num_column_colors1) = color_stats[problem][mode][:row_colors]
                (num_row_colors2, num_column_colors2) = color_stats[problem][mode][:all_colors]
                (num_row_colors3, num_column_colors3) = color_stats[problem][mode][:column_colors]
                @test num_row_colors1 ≤ num_row_colors2 ≤ num_row_colors3
                @test num_column_colors3 ≤ num_column_colors2 ≤ num_column_colors1
            end
        end
    end
end;
