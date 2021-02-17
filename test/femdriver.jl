module FEMDriver

using Test
using Gridap
using GridapPETSc

tol = 1e-10

GridapPETSc.Init(["-ksp_rtol","$tol"])

domain = (0,1,0,1,0,1)
cells  = (10,10,10)
model  = CartesianDiscreteModel(domain,cells)

order = 1
V = TestFESpace( model,
      ReferenceFE(lagrangian,Float64,order),
      conformity=:H1, dirichlet_tags="boundary" )
U = TrialFESpace(V)

Ω = Triangulation(model)

degree = 2*order
dΩ = Measure(Ω,degree)

f(x) = x[1]*x[2]

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
l(v) = ∫( v*f )*dΩ

ass = SparseMatrixAssembler(SparseMatrixCSR{0,PetscReal,PetscInt},U,V)
op = AffineFEOperator(a,l,ass)

ls = PETScSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

GridapPETSc.Finalize()

end #module
