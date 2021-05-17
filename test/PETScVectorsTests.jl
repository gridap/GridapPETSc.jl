module PETScVectorsTests

using GridapPETSc
using Test
using GridapPETSc: PetscScalar

options = "-info"
GridapPETSc.Init(args=split(options))

n = 10
v = PETScVector(n)

@test length(v) == n
@test size(v) == (n,)
display(axes(v))
display(eachindex(v))
display(LinearIndices(v))
display(CartesianIndices(v))

v[0] = 1

display(v)

s = similar(v,PetscScalar,4)
display(s)
w = similar(v)
display(w)

# Objects need to be out of scope or finalized
# before calling GridapPETSc.Finalize()
w = nothing
GridapPETSc.Finalize(s)
v = nothing

GridapPETSc.Finalize()

end # module
