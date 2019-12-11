module NonSymFEMDriver

using Test
using Gridap
using GridapPETSc
using SparseMatricesCSR

tol = 1e-10

GridapPETSc.init!(["-ksp_rtol","$tol"]) 

model = CartesianDiscreteModel(
  domain=(0.0,1.0,0.0,1.0), partition=(2,2))

const T = VectorValue{2,Float64}

order = 2
fespace1 = FESpace(
  reffe = :QLagrangian,
  conformity = :H1,
  valuetype = T,
  model = model,
  order = order,
  diritags = "boundary")

fespace2 = FESpace(
  reffe = :PLagrangian,
  conformity = :L2,
  valuetype = Float64,
  model = model,
  order = order-1)

fespace2 = ConstrainedFESpace(fespace2,[1,])

V = TestFESpace(fespace1)
Q = TestFESpace(fespace2)
Y = [V, Q]

U = TrialFESpace(fespace1, zero(T))
P = TrialFESpace(fespace2)
X = [U, P]

trian = Triangulation(model)
quad = CellQuadrature(trian,degree=4)

function a_Ω(y,x)
  v, q = y
  u, p = x
  inner(∇(v),∇(u)) - inner(div(v),p) + inner(q,div(u))
end

function b_Ω(y)
  v, q = y
  inner(v,(x)->2*x)
end

t_Ω = AffineFETerm(a_Ω,b_Ω,trian,quad)

op = LinearFEOperator(SymSparseMatrixCSR{0,PetscReal,PetscInt},Y,X,t_Ω)

ls = PETScSolver()
# This one works: ls = LUSolver()
solver = LinearFESolver(ls)

xh = solve(solver,op)

@show num_free_dofs(U)
@show num_free_dofs(P)

x = free_dofs(xh)
A = op.mat
b = op.vec

@show all(abs.(A - A') .< tol)

r = A*x - b

@test maximum(abs.(r))/maximum(abs.(b)) < tol

GridapPETSc.finalize!()

end # module
