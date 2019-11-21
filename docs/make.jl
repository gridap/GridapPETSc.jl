using Documenter, GridapPETSc

makedocs(;
    modules=[GridapPETSc],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gridap/GridapPETSc.jl/blob/{commit}{path}#L{line}",
    sitename="GridapPETSc.jl",
    authors="Víctor Sande <vsande@cimne.upc.edu>",
    assets=String[],
)

deploydocs(;
    repo="github.com/gridap/GridapPETSc.jl",
)
