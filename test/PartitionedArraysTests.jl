using Gridap
using GridapPETSc
using GridapPETSc.PETSC
using PartitionedArrays
using SparseMatricesCSR
using Test
using LinearAlgebra

using PartitionedArrays: assemble_coo!

function set_mumps_options(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  # percentage increase in the estimated working space
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  14, 1000)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end

function partitioned_tests(distribute,nparts)
  parts = distribute(LinearIndices((prod(nparts),)))

  ngids = 7
  if length(parts) > 1

    ids_partition = map(parts) do part
      if part == 1
        own   = OwnIndices(ngids,1,[1,2,3])
        ghost = GhostIndices(ngids,[5,6],[2,3])
        OwnAndGhostIndices(own,ghost)
      elseif part == 2
        own   = OwnIndices(ngids,2,[4,5])
        ghost = GhostIndices(ngids,[7,3,6],[3,1,3])
        OwnAndGhostIndices(own,ghost)
      elseif part == 3
        own   = OwnIndices(ngids,3,[6,7])
        ghost = GhostIndices(ngids,[4],[2])
        OwnAndGhostIndices(own,ghost)
      end
    end

    I,J,V = map(parts) do part
      if part == 1
        I = [1,2,3,1,3]
        J = [1,2,3,5,6]
        V = [9,9,9,1,1]
      elseif part == 2
        I = [4,5,5,4,5]
        J = [4,5,3,6,7]
        V = [9,9,9,1,1]
      elseif part == 3
        I = [6,7,6]
        J = [6,7,4]
        V = [9,9,1]
      end
      I,J,Float64.(V)
    end |> tuple_of_arrays

  else

    ids_partition = map(parts) do part
      own   = OwnIndices(ngids,part,collect(1:ngids))
      ghost = GhostIndices(ngids,Int[],Int32[])
      OwnAndGhostIndices(own,ghost)
    end

    I,J,V = map(parts) do part
      I = [1,2,3,1,3,4,5,5,4,5,6,7,6]
      J = [1,2,3,5,6,4,5,3,6,7,6,7,4]
      V = [9,9,9,1,1,9,9,1,1,1,9,9,1]
      I,J,Float64.(V)
    end |> tuple_of_arrays

  end

  GridapPETSc.Init(args=split("-ksp_type gmres -ksp_converged_reason -ksp_error_if_not_converged true -pc_type jacobi -ksp_rtol 1.0e-12"))

  function test_get_local_vector(v::PVector,x::PETScVector)
    if isa(partition(v),MPIArray)
      map(parts) do part
        lg = GridapPETSc._get_local_oh_vector(x.vec[])
        @test isa(lg,PETScVector)
        lx = GridapPETSc._get_local_vector(lg)
        if part==1
          @test length(lx)==5
        elseif part==2
          @test length(lx)==5
        elseif part==3
          @test length(lx)==3
        end
        GridapPETSc._restore_local_vector!(lx,lg)
        GridapPETSc.Finalize(lg)
      end
    end
  end

  function test_vectors(v::PVector,x::PETScVector,ids)
    PETSC.@check_error_code PETSC.VecView(x.vec[],C_NULL)
    u = PVector(x,ids)
    consistent!(v) |> fetch
    consistent!(u) |> fetch
    map(partition(u),partition(v)) do u,v
      @test u == v
    end
  end

  ids = PRange(ids_partition)
  values = map(partition(ids)) do ids
    println(local_to_global(ids))
    return 10.0*local_to_global(ids)
  end

  v = PVector(values,partition(ids))
  x = convert(PETScVector,v)
  test_get_local_vector(v,x)
  test_vectors(v,x,ids)

  if isa(partition(v),MPIArray)
    # Copy v into v1 to circumvent (potentia) aliasing of v and x
    v1 = copy(v)
    fill!(v1,zero(eltype(v)))
    copy!(v1,x)
    consistent!(v1) |> fetch
    test_vectors(v1,x,ids)

    # Copy x into x1 to circumvent (potential) aliasing of v and x
    x1 = copy(x)
    fill!(x1,PetscScalar(0.0))
    copy!(x1,v)
    test_vectors(v,x1,ids)
    GridapPETSc.Finalize(x1)
  end

  A = psparse!(I,J,V,partition(ids),partition(ids);discover_rows=false,discover_cols=false) |> fetch
  display(partition(A))
  B = convert(PETScMatrix,A)
  PETSC.@check_error_code PETSC.MatView(B.mat[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)

  function solve_system_and_check_solution(A::PSparseMatrix,B::PETScMatrix,v)
    solver = PETScLinearSolver()
    ss = symbolic_setup(solver,A)
    ns = numerical_setup(ss,A)
    consistent!(v) |> fetch
    y = pfill(0.0,partition(ids))
    z = pfill(0.0,partition(ids))
    mul!(y,A,v)
    consistent!(y) |> fetch
    z = solve!(z,ns,y)
    consistent!(z) |> fetch

    nspetsc = numerical_setup(symbolic_setup(PETScLinearSolver(),B),B)
    ypetsc = convert(PETScVector,y)
    zpetsc = PETScVector(0.0,ids)
    zpetsc = solve!(zpetsc,nspetsc,ypetsc)

    test_vectors(y,ypetsc,ids)
    test_vectors(z,zpetsc,ids)
    
    map(parts,partition(z),partition(v)) do p,z,v
      @test maximum(abs.(z-v)) < 1e-5
    end
    GridapPETSc.Finalize(ypetsc)
    GridapPETSc.Finalize(zpetsc)
  end
  solve_system_and_check_solution(A,B,v)

  # Test that copy! works ok
  LinearAlgebra.fillstored!(B,PetscScalar(0.0))
  copy!(B,A)
  solve_system_and_check_solution(A,B,v)

  GridapPETSc.Finalize(B)
  GridapPETSc.Finalize(x)
  GridapPETSc.Finalize()
end
