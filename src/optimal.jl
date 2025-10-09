"""
    OptimalColoringAlgorithm

Coloring algorithm that relies on mathematical programming with [JuMP](https://jump.dev/) to find an optimal coloring.

!!! warning
    This algorithm is only available when JuMP is loaded. If you encounter a method error, run `import JuMP` in your REPL and try again.

# Constructor

    OptimalColoringAlgorithm(optimizer)

The `optimizer` argument can be any JuMP-compatible optimizer, like `HiGHS.Optimizer`.
You can use [`optimizer_with_attributes`](https://jump.dev/JuMP.jl/stable/api/JuMP/#optimizer_with_attributes) to set solver-specific parameters.
"""
struct OptimalColoringAlgorithm{S} <: ADTypes.AbstractColoringAlgorithm
    optimizer::S
end
