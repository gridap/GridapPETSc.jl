using Documenter, GridapPETSc

makedocs(;
    modules=[GridapPETSc],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Julia interfaces" => "library.md",
        "PETSc wrappers" => "wrappers.md",
    ],
    repo="https://github.com/gridap/GridapPETSc.jl/blob/{commit}{path}#L{line}",
    sitename="GridapPETSc",
    authors="Francesc Verdugo <fverdugo@cimne.upc.edu> and Víctor Sande <vsande@cimne.upc.edu>"
)

deploydocs(;
    repo="github.com/gridap/GridapPETSc.jl",
)
