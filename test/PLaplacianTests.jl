using Gridap
using Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using Test


function mysnessetup(snes)
  ksp      = Ref{GridapPETSc.PETSC.KSP}()
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.SNESSetFromOptions(snes[])
  @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCView(pc[],C_NULL)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end


function main(parts)
  options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 1.0 -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -snes_converged_reason"

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

  nls = PETScNonlinearSolver(mysnessetup)
  solver = FESolver(nls)
  uh = solve(solver,op)

  Ωo = Triangulation(model)
  dΩo = Measure(Ωo,2*k)
  eh = u - uh
  @test sqrt(sum(∫( abs2(eh) )dΩo)) < 1.0e-9

end
