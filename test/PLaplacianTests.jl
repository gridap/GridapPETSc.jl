using Gridap
using Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using Test

function main(parts)
  options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 1.0 -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -pc_type jacobi -ksp_type gmres -ksp_monitor -snes_converged_reason"

  GridapPETSc.with(args=split(options)) do
     main(parts,FullyAssembledRows())
     main(parts,SubAssembledRows())
  end
end

function main(parts,strategy)

  domain = (0,4,0,4)
  cells = (100,100)
  model = CartesianDiscreteModel(parts,domain,cells)

  k = 1
  u((x,y)) = (x+y)^k
  σ(∇u) =(1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = 2*∇u⋅∇du + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  Ω = Triangulation(strategy,model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  op = FEOperator(r,j,U,V,strategy)

  uh = zero(U)
  # b,A = residual_and_jacobian(op,uh)
  # _A = copy(A)
  # _b = copy(b)
  # residual_and_jacobian!(_b,_A,op,uh)
  # @test (norm(_b-b)+1) ≈ 1
  # x = similar(b,Float64,axes(A,2))
  # fill!(x,1)
  # @test (norm(A*x-_A*x)+1) ≈ 1

  nls = PETScNonlinearSolver()
  solver = FESolver(nls)
  uh = solve(solver,op)

  Ωo = Triangulation(model)
  dΩo = Measure(Ωo,2*k)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩo)) < 1.0e-9

end
