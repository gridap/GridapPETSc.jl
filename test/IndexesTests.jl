using GridapPETSc
using Test
using SparseArrays
using SparseMatricesCSR
using GridapPETSc: PetscScalar, PetscInt
using LinearAlgebra
using MPI
using PartitionedArrays


function main(parts)
options = "-info"
  GridapPETSc.with(args=split(options)) do
    backend = get_backend(parts)
    if backend == MPIBackend()
      comm = MPI.COMM_WORLD
      procid = PartitionedArrays.get_part_id(comm)
      nprocs = PartitionedArrays.num_parts(comm)
    elseif backend == SequentialBackend()
      procid = 1
      nprocs = 1
    end

    println(procid)
    array = ones(5)
    array = array .+ procid
    is =  PETScIS(array)
    @check_error_code GridapPETSc.PETSC.ISView(is.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD)
    @test is.size[1] == length(array)*nprocs
  end
end

#mpiexecjl --project=. -n 4 julia test/IndexesTests.jl