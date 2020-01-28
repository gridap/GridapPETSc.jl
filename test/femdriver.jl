module FEMDriver

using Test
using Gridap
using GridapPETSc

tol = 1e-10

GridapPETSc.init!(["-ksp_rtol","$tol"]) 

model = CartesianDiscreteModel((0,1,0,1,0,1), (10,10,10))

V = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

U = TrialFESpace(V)

trian = get_triangulation(model)
quad = CellQuadrature(trian,2)

t_Ω = AffineFETerm(
  (v,u) -> inner(∇(v),∇(u)),
  (v) -> inner(v, (x) -> x[1]*x[2] ),
  trian, quad)

#op = AffineFEOperator(SparseMatrixCSR{0,PetscReal,PetscInt},V,U,t_Ω)
op = AffineFEOperator(V,U,t_Ω)

ls = PETScSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = op.op.matrix
b = op.op.vector

r = A*x - b
@test maximum(abs.(r)) < tol

GridapPETSc.finalize!()

end #module
