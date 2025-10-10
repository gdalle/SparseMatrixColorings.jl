"""
    OptimalColoringAlgorithm

Coloring algorithm that relies on mathematical programming with [JuMP](https://jump.dev/) to find an optimal coloring.

!!! warning
    This algorithm is only available when JuMP is loaded. If you encounter a method error, run `import JuMP` in your REPL and try again.

!!! danger
    The coloring problem is NP-hard, so it is unreasonable to expect an optimal solution in reasonable time for large instances.

# Constructor

    OptimalColoringAlgorithm(optimizer; silent::Bool=true)

The `optimizer` argument can be any JuMP-compatible optimizer.
However, the problem formulation is best suited to CP-SAT optimizers like [MiniZinc](https://github.com/jump-dev/MiniZinc.jl).
You can use [`optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to set solver-specific parameters.
"""
struct OptimalColoringAlgorithm{O} <: ADTypes.AbstractColoringAlgorithm
    optimizer::O
    silent::Bool
end

function OptimalColoringAlgorithm(optimizer; silent::Bool=true)
    return OptimalColoringAlgorithm(optimizer, silent)
end
