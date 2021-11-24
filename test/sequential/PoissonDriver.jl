module PoissonDriver

using Test
using Gridap
using Gridap.FESpaces
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt
using SparseArrays
using SparseMatricesCSR

# GridapDistributedPETScWrappers.C.KSPCreate(comm(A),ksp)
# GridapDistributedPETScWrappers.C.KSPSetOperators(ksp[],A.p,A.p)
# GridapDistributedPETScWrappers.C.KSPSetType(ksp[],GridapDistributedPETScWrappers.C.KSPPREONLY)
# GridapDistributedPETScWrappers.C.KSPGetPC(ksp[],pc)

# # If system is SPD use the following two calls
# GridapDistributedPETScWrappers.C.PCSetType(pc[],GridapDistributedPETScWrappers.C.PCCHOLESKY)
# GridapDistributedPETScWrappers.C.MatSetOption(A.p,
#                                               GridapDistributedPETScWrappers.C.MAT_SPD,GridapDistributedPETScWrappers.C.PETSC_TRUE);
# # Else ... use only the following one
# # GridapDistributedPETScWrappers.C.PCSetType(pc,GridapDistributedPETScWrappers.C.PCLU)

# PCFactorSetMatSolverType(pc[],GridapDistributedPETScWrappers.C.MATSOLVERMUMPS)
# PCFactorSetUpMatSolverType(pc[])
# GridapDistributedPETScWrappers.C.PCFactorGetMatrix(pc[],mumpsmat)
# MatMumpsSetIcntl(mumpsmat[],4 ,2)     # level of printing (0 to 4)
# MatMumpsSetIcntl(mumpsmat[],28,2)     # use 1 for sequential analysis and ictnl(7) ordering,
#                                     # or 2 for parallel analysis and ictnl(29) ordering
# MatMumpsSetIcntl(mumpsmat[],29,2)     # parallel ordering 1 = ptscotch, 2 = parmetis
# MatMumpsSetCntl(mumpsmat[] ,3,1.0e-6)  # threshhold for row pivot detection


tol = 1e-10
maxits = 1000
options = [
  "-ksp_type", "cg",
  "-ksp_monitor",
  "-ksp_rtol", "$tol",
  "-ksp_converged_reason",
  "-ksp_max_it", "$maxits",
  "-ksp_norm_type", "unpreconditioned",
  "-ksp_view",
  "-pc_type","gamg",
  "-pc_gamg_type","agg",
  "-mg_levels_esteig_ksp_type","cg",
  "-mg_coarse_sub_pc_type","cholesky",
  "-mg_coarse_sub_pc_factor_mat_ordering_type","nd",
  "-pc_gamg_process_eq_limit","50",
  "-pc_gamg_square_graph","0",
  "-pc_gamg_agg_nsmooths","1"]

# Safely open and close PETSc
GridapPETSc.with(args=options) do

  domain = (0,1,0,1,0,1)
  cells  = (10,10,10)
  model  = CartesianDiscreteModel(domain,cells)

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary",vector_type=Vector{PetscScalar})
  U = TrialFESpace(V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)

  f(x) = x[1]*x[2]
  a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
  l(v) = ∫( v*f )*dΩ

  solver = LinearFESolver(PETScLinearSolver())

  # Assembling on a Julia matrix
  # with the same data layout as petsc
  # (efficient use of PETScLinearSolver)
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  Tx = get_vector_type(U)
  assem = SparseMatrixAssembler(Tm,Tv,U,V)
  op = AffineFEOperator(a,l,U,V,assem)
  uh = solve(solver,op)
  x = get_free_dof_values(uh)
  A = get_matrix(op)
  b = get_vector(op)
  @test typeof(A) == Tm
  @test typeof(b) == Tv
  @test typeof(x) == Tx
  r = A*x - b
  @test maximum(abs.(r)) < tol

  # Assembling on a Julia matrix
  # with different data layout than petsc
  # (inefficient use of PETScLinearSolver)
  Tm = SparseMatrixCSC{Float64,Int}
  Tv = Vector{Float64}
  Tx = get_vector_type(U)
  assem = SparseMatrixAssembler(Tm,Tv,U,V)
  op = AffineFEOperator(a,l,U,V,assem)
  uh = solve(solver,op)
  x = get_free_dof_values(uh)
  A = get_matrix(op)
  b = get_vector(op)
  @test typeof(A) == Tm
  @test typeof(b) == Tv
  @test typeof(x) == Tx
  r = A*x - b
  @test maximum(abs.(r)) < tol

  # Now assemble on a native PETScMarix but on a Julia Vector
  # with same memory layout as PETScVector
  # (efficient use of PETScLinearSolver)
  Tm = PETScMatrix
  Tv = Vector{PetscScalar}
  Tx = get_vector_type(U)
  assem = SparseMatrixAssembler(Tm,Tv,U,V)
  op = AffineFEOperator(a,l,U,V,assem)
  uh = solve(solver,op)
  x = get_free_dof_values(uh)
  A = get_matrix(op)
  b = get_vector(op)
  @test typeof(A) == Tm
  @test typeof(b) == Tv
  @test typeof(x) == Tx
  r = A*x - b
  @test maximum(abs.(r)) < tol

  # Now assemble on a native PETScMarix and on a native PETScVector
  # (efficient use of PETScLinearSolver)
  Tm = PETScMatrix
  Tv = PETScVector
  Tx = get_vector_type(U)
  assem = SparseMatrixAssembler(Tm,Tv,U,V)
  op = AffineFEOperator(a,l,U,V,assem)
  uh = solve(solver,op)
  x = get_free_dof_values(uh)
  A = get_matrix(op)
  b = get_vector(op)
  @test typeof(A) == Tm
  @test typeof(b) == Tv
  @test typeof(x) == Tx
  r = A*x - b
  @test maximum(abs.(r)) < tol

end

end # module
