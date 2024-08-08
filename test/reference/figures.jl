"""
    what_fig_41()

Return the symmetric matrix from Figure 4.1 of "What color is your Jacobian?", filled with unique values.
"""
function what_fig_41()
    #! format: off
    A = [
        1  2  0  0  0  0
        2  3  4  0  5  6
        0  4  7  8  0  0
        0  0  8  9  0  10
        0  5  0  0  11 0
        0  6  0  10 0  12
    ]
    #! format: on
    @assert A == transpose(A)
    return sparse(A)
end

"""
    efficient_fig_1()

Return the symmetric matrix from Figure 1 of "Efficient computation of sparse hessians using coloring and AD", filled with unique values 
"""
function efficient_fig_1()
    #! format: off
    A = [
        1  2  0  0  0  0  3  0  0  0
        2  4  5  0  6  0  0  0  0  0
        0  5  7  8  0  9  0  0  0  0
        0  0  8  10 0  0  0  0  0  11
        0  6  0  0  12 13 0  14 0  0
        0  0  9  0  13 15 0  0  16 0
        3  0  0  0  0  0  17 18 0  0
        0  0  0  0  14 0  18 19 20 0
        0  0  0  0  0  16 0  20 21 22
        0  0  0  11 0  0  0  0  22 23
    ]
    #! format: on
    @assert A == transpose(A)
    return sparse(A)
end
