module PLaplacianDriver

using Test
using Gridap
using Gridap.FESpaces
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt
using SparseArrays
using SparseMatricesCSR

options = "-snes_type newtonls -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -ksp_converged_reason -pc_type ilu -ksp_type gmres -snes_converged_reason"

msg = "some output"

out = GridapPETSc.with(args=split(options)) do

  domain = (0,4,0,4)
  cells = (100,100)
  model = CartesianDiscreteModel(domain,cells)

  k = 1
  u((x,y)) = (x+y)^k
  σ(∇u) =(1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = 2*∇u⋅∇du + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  nls = PETScNonlinearSolver()

  op = FEOperator(r,j,U,V)
  uh = solve(nls,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  assem = SparseMatrixAssembler(Tm,Tv,U,V)
  op = FEOperator(r,j,U,V,assem)
  uh = solve(nls,op)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

  msg
end

@test out === msg

end # module
