module PoissonDriver

using Test
using Gridap
using Gridap.FESpaces
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt
using SparseArrays
using SparseMatricesCSR

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

  solver = LinearFESolver(PETScSolver())
  
  # Assembling on a Julia matrix 
  # with the same data layout as petsc
  # (efficient use of PETScSolver)
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
  # (inefficient use of PETScSolver)
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
  # (efficient use of PETScSolver)
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
  # (efficient use of PETScSolver)
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
