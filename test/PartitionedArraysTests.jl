using Gridap
using GridapPETSc
using GridapPETSc.PETSC
using PartitionedArrays
using SparseMatricesCSR
using Test

function partitioned_tests(parts)

  if length(parts) > 1

    noids,firstgid,hid_to_gid,hid_to_part = map_parts(parts) do part
      if part == 1
        3,1,[5,6],Int32[2,3]
      elseif part == 2
        2,4,[7,3,6],Int32[3,1,3]
      elseif part == 3
        2,6,[4],Int32[2]
      end
    end

    I,J,V = map_parts(parts) do part
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
    end

  else

    noids,firstgid,hid_to_gid,hid_to_part = map_parts(parts) do part
      7,1,Int[],Int32[]
    end

    I,J,V = map_parts(parts) do part
      I = [1,2,3,1,3,4,5,5,4,5,6,7,6]
      J = [1,2,3,5,6,4,5,3,6,7,6,7,4]
      V = [9,9,9,1,1,9,9,1,1,1,9,9,1]
      I,J,Float64.(V)
    end

  end

  GridapPETSc.Init(args=split("-ksp_type gmres -ksp_monitor -pc_type jacobi"))

  ngids = 7
  ids = PRange(parts,ngids,noids,firstgid,hid_to_gid,hid_to_part)
  values = map_parts(ids.partition) do ids
    10.0*ids.lid_to_gid
  end
  v = PVector(values,ids)
  x = PETScVector(v)
  if (length(parts)>1)
    map_parts(parts) do part
      lg=get_local_oh_vector(x)
      @test isa(lg,PETScVector)
      lx=get_local_vector(lg)
      if part==1
        @test length(lx)==5
      elseif part==2
        @test length(lx)==5
      elseif part==3
        @test length(lx)==3
      end
      restore_local_vector!(lx,lg)
      GridapPETSc.Finalize(lg)
    end
  end

  PETSC.@check_error_code PETSC.VecView(x.vec[],C_NULL)
  u = PVector(x,ids)
  exchange!(u)
  map_parts(u.values,v.values) do u,v
    @test u == v
  end

  A = PSparseMatrix(I,J,V,ids,ids,ids=:global)
  display(A.values)
  B = PETScMatrix(A)
  PETSC.@check_error_code PETSC.MatView(B.mat[],PETSC.@PETSC_VIEWER_STDOUT_WORLD)

  #TODO hide conversions and solver setup
  solver = PETScLinearSolver()
  ss = symbolic_setup(solver,B)
  ns = numerical_setup(ss,B)
  y = PETScVector(A*v)
  x̂ = PETScVector(0.0,ids)
  solve!(x̂,ns,y)
  z = PVector(x̂,ids)
  exchange!(z)
  map_parts(z.values,v.values) do z,v
    @test maximum(abs.(z-v)) < 1e-5
  end

  GridapPETSc.Finalize(x̂)
  GridapPETSc.Finalize(B)
  GridapPETSc.Finalize(y)
  GridapPETSc.Finalize(ns)
  GridapPETSc.Finalize(x)
  GridapPETSc.Finalize()
end
