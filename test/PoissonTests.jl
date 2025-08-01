using SparseMatricesCSR
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using GridapPETSc.PETSC
using PartitionedArrays
using Test

# Setup solver via low level PETSC API calls
function mykspsetup(ksp)
  pc       = Ref{PETSC.PC}()
  mumpsmat = Ref{PETSC.Mat}()
  @check_error_code PETSC.KSPSetType(ksp[],PETSC.KSPPREONLY)
  @check_error_code PETSC.KSPGetPC(ksp[],pc)
  @check_error_code PETSC.PCSetType(pc[],PETSC.PCLU)
  @check_error_code PETSC.PCFactorSetMatSolverType(pc[],PETSC.MATSOLVERMUMPS)
  @check_error_code PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 2)
  @check_error_code PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
  @check_error_code PETSC.KSPSetFromOptions(ksp[])
end

function main(distribute,nparts)
  main(distribute,nparts,:cg)
  if GridapPETSc.libpetsc_provider === "JULIA_PETSC_LIBRARY"
    main(distribute,nparts,:mumps)
  end
end

function main(distribute,nparts,solver)
  parts = distribute(LinearIndices((prod(nparts),)))

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
      model = CartesianDiscreteModel(parts,nparts,domain,cells)

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

      Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
      Tv = Vector{PetscScalar}
      assem = SparseMatrixAssembler(Tm,Tv,U,V)
      op = AffineFEOperator(a,l,U,V,assem)

      v_julia = get_vector(op)
      v_petsc = convert(PETScVector,v_julia)
      copy!(v_julia,v_petsc)
      copy!(v_petsc,v_julia)
      GridapPETSc.destroy(v_petsc)

      # Checking that convert performs deep copies and does not modify A
      A = get_matrix(op)
      vals_copy = map(partition(A)) do A
        @test typeof(A)==SparseMatrixCSR{0,PetscScalar,PetscInt}
        i = copy(A.rowptr)
        j = copy(A.colval)
        a = copy(A.nzval)
        i,j,a
      end
      Apetsc = convert(PETScMatrix,A)
      map(partition(A),vals_copy) do A, (i,j,a)
        @test all(i .== A.rowptr)
        @test all(j .== A.colval)
        @test all(a .== A.nzval)
      end
      GridapPETSc.destroy(Apetsc)

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
