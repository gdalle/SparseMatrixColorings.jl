"""
$TYPEDEF

Example coloring problem from one of our reference articles.

Used for internal testing.

# Fields

$TYPEDFIELDS
"""
struct Example{TA<:AbstractMatrix,TB<:AbstractMatrix}
    "decompressed matrix"
    A::TA
    "column-compressed matrix"
    B::TB
    "vector of colors"
    color::Vector{Int}
end

"""
    what_fig_41()

Construct an [`Example`](@ref) from Figure 4.1 of "What color is your Jacobian?", where the nonzero entries are filled with unique values.
"""
function what_fig_41()
    #! format: off
    M = sparse([
    #   1 2 3 4 5 6
        1 1 0 0 0 0  # 1
        1 1 1 0 1 1  # 2
        0 1 1 1 0 0  # 3
        0 0 1 1 0 1  # 4
        0 1 0 0 1 0  # 5
        0 1 0 1 0 1  # 6
    ])
    #! format: on
    @assert M == transpose(M)
    nonzeros(M) .= 1:length(nonzeros(M))
    A = sparse(Symmetric(M))
    color = [
        1,  # 1. green
        2,  # 2. red
        1,  # 3. green
        3,  # 4. blue
        1,  # 5. green
        1,  # 6. green
    ]
    B = hcat(
        A[:, 1] .+ A[:, 3] .+ A[:, 5] .+ A[:, 6],  # green
        A[:, 2],  # red
        A[:, 4],  # blue
    )
    return Example(A, B, color)
end

"""
    what_fig_61()

Construct an [`Example`](@ref) from Figure 6.1 of "What color is your Jacobian?", where the nonzero entries are filled with unique values.
"""
function what_fig_61()
    #! format: off
    M = sparse([
    #   1 2 3 4 5 6 7 8 9 10
        1 1 0 0 0 0 1 0 0 0  # 1
        1 1 1 0 1 0 0 0 0 0  # 2
        0 1 1 1 0 1 0 0 0 0  # 3
        0 0 1 1 0 0 0 0 0 1  # 4
        0 1 0 0 1 1 0 1 0 0  # 5
        0 0 1 0 1 1 0 0 1 0  # 6
        1 0 0 0 0 0 1 1 0 0  # 7
        0 0 0 0 1 0 1 1 1 0  # 8
        0 0 0 0 0 1 0 1 1 1  # 9
        0 0 0 1 0 0 0 0 1 1  # 10
    ])
    #! format: on
    @assert M == transpose(M)
    nonzeros(M) .= 1:length(nonzeros(M))
    A = sparse(Symmetric(M))
    color = [
        1,  # 1. red
        2,  # 2. blue
        1,  # 3. red
        2,  # 4. blue
        1,  # 5. red
        3,  # 6. green
        2,  # 7. blue
        3,  # 8. green
        2,  # 9. blue
        3,  # 10. green
    ]
    B = hcat(
        A[:, 1] .+ A[:, 3] .+ A[:, 5],  # red
        A[:, 2] .+ A[:, 4] .+ A[:, 7] .+ A[:, 9],  # blue
        A[:, 6] .+ A[:, 8] .+ A[:, 10],  # green
    )
    return Example(A, B, color)
end

"""
    efficient_fig_1()

Construct an [`Example`](@ref) from Figure 1 of "Efficient computation of sparse hessians using coloring and AD", where the nonzero entries are filled with unique values.
"""
function efficient_fig_1()
    #! format: off
    M = sparse([
    #   1 2 3 4 5 6 7 8 9 10    
        1 1 0 0 0 0 1 0 0 0  # 1
        1 1 1 0 1 0 0 0 0 0  # 2
        0 1 1 1 0 1 0 0 0 0  # 3
        0 0 1 1 0 0 0 0 0 1  # 4
        0 1 0 0 1 1 0 1 0 0  # 5
        0 0 1 0 1 1 0 0 1 0  # 6
        1 0 0 0 0 0 1 1 0 0  # 7
        0 0 0 0 1 0 1 1 1 0  # 8
        0 0 0 0 0 1 0 1 1 1  # 9
        0 0 0 1 0 0 0 0 1 1  # 10
    ])
    #! format: on
    @assert M == transpose(M)
    nonzeros(M) .= 1:length(nonzeros(M))
    A = sparse(Symmetric(M))
    color = [
        1,  # 1. red
        2,  # 2. cyan
        1,  # 3. red
        3,  # 4. yellow
        1,  # 5. red
        4,  # 6. green
        3,  # 7. yellow
        5,  # 8. navy blue
        1,  # 9. red
        2,  # 10. cyan
    ]
    B = hcat(
        A[:, 1] .+ A[:, 3] .+ A[:, 5] .+ A[:, 9], # red
        A[:, 2] .+ A[:, 10],  # cyan
        A[:, 4] .+ A[:, 7],  # yellow
        A[:, 6],  # green
        A[:, 8],  # navy blue
    )
    return Example(A, B, color)
end

"""
    efficient_fig_4()

Construct an [`Example`](@ref) from Figure 4 of "Efficient computation of sparse hessians using coloring and AD", where the nonzero entries are filled with unique values.
"""
function efficient_fig_4()
    #! format: off
    M = sparse([
    #   1 2 3 4 5 6 7 8 9 10    
        1 1 0 0 0 0 1 0 0 0  # 1
        1 1 1 0 1 0 0 0 0 0  # 2
        0 1 1 1 0 1 0 0 0 0  # 3
        0 0 1 1 0 0 0 0 0 1  # 4
        0 1 0 0 1 1 0 1 0 0  # 5
        0 0 1 0 1 1 0 0 1 0  # 6
        1 0 0 0 0 0 1 1 0 0  # 7
        0 0 0 0 1 0 1 1 1 0  # 8
        0 0 0 0 0 1 0 1 1 1  # 9
        0 0 0 1 0 0 0 0 1 1  # 10
    ])
    #! format: on
    @assert M == transpose(M)
    nonzeros(M) .= 1:length(nonzeros(M))
    A = sparse(Symmetric(M))
    color = [
        1,  # 1. red
        2,  # 2. cyan
        1,  # 3. red
        2,  # 4. cyan
        1,  # 5. red
        3,  # 6. yellow
        2,  # 7. cyan
        3,  # 8. yellow
        2,  # 9. cyan
        1,  # 10. red
    ]
    B = hcat(
        A[:, 1] .+ A[:, 3] .+ A[:, 5] .+ A[:, 10], # red
        A[:, 2] .+ A[:, 4] .+ A[:, 7] .+ A[:, 9],  # cyan
        A[:, 6] .+ A[:, 8],  # yellow
    )
    return Example(A, B, color)
end
