using Documenter
using DocumenterInterLinks
using SparseMatrixColorings

links = InterLinks(
    "ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/",
    "BandedMatrices" => "https://julialinearalgebra.github.io/BandedMatrices.jl/stable/",
    "LinearAlgebra" => "https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/",
)

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[SparseMatrixColorings],
    authors="Guillaume Dalle and Alexis Montoison",
    sitename="SparseMatrixColorings.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "tutorial.md",
        "api.md",
        "Developer Documentation" => ["dev.md", "vis.md"],
    ],
    plugins=[links],
)

deploydocs(;
    repo="github.com/gdalle/SparseMatrixColorings.jl", push_preview=true, devbranch="main"
)
