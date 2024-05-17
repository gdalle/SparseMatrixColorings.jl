using Documenter
using SparseMatrixColorings

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[SparseMatrixColorings],
    authors="Guillaume Dalle",
    sitename="SparseMatrixColorings.jl",
    format=Documenter.HTML(),
    pages=["Home" => "index.md", "API reference" => "api.md"],
)

deploydocs(; repo="github.com/gdalle/SparseMatrixColorings.jl", devbranch="main")
