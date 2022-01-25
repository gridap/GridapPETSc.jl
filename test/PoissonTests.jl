using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using GridapPETSc: PETSC
using PartitionedArrays
using Test

# Setup solver via low level PETSC API calls
function mykspsetup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
end

function main(parts)
  main(parts,:cg)
  if PETSC.MatMumpsSetIcntl_handle[] != C_NULL
    main(parts,:mumps)
  end
end

function main(parts,solver)
  if solver == :mumps
    options = "-ksp_error_if_not_converged true -ksp_converged_reason"
  elseif solver == :cg
    options = "-pc_type jacobi -ksp_type cg -ksp_error_if_not_converged true -ksp_converged_reason -ksp_rtol 1.0e-12"
  else
    error()
  end
  GridapPETSc.with(args=split(options)) do
      domain = (0,4,0,4)
      cells = (4,4)
      model = CartesianDiscreteModel(parts,domain,cells)

      labels = get_face_labeling(model)
      add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
      add_tag_from_tags!(labels,"neumann",[4,6,8])

      Ω = Triangulation(model)
      Γn = Boundary(model,tags="neumann")
      n_Γn = get_normal_vector(Γn)

      k = 2
      u((x,y)) = (x+y)^k
      f(x) = -Δ(u,x)
      g = n_Γn⋅∇(u)

      reffe = ReferenceFE(lagrangian,Float64,k)
      V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
      U = TrialFESpace(u,V)

      dΩ = Measure(Ω,2*k)
      dΓn = Measure(Γn,2*k)

      a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
      l(v) = ∫( v*f )dΩ + ∫( v*g )dΓn
      assem=SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{Float64},U,V)
      op = AffineFEOperator(a,l,U,V,assem)

      if solver == :mumps
        ls = PETScLinearSolver(mykspsetup)
      else
        ls = PETScLinearSolver()
      end
      fels = LinearFESolver(ls)
      uh = solve(fels,op)
      eh = u - uh
      @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
  end
 end
