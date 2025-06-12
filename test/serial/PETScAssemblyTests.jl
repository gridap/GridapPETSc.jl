module PETScAssemblyTests

using Test
using Gridap
using Gridap.Algebra
using Gridap.Arrays
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt
using SparseArrays
using LinearAlgebra

options = "-info"

GridapPETSc.with(args=split(options)) do

  touch! = TouchEntriesMap()
  add! = AddEntriesMap(+)

  Tm = PETScMatrix
  builder = SparseMatrixBuilder(Tm)
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(a,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  b = nz_allocation(a)
  add_entries!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(b,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  c = create_from_nz(b)
  add_entries!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(c,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  d = create_from_nz(c)
  display(d)

  builder = SparseMatrixBuilder(Tm,MinMemory(3))
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(a,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  b = nz_allocation(a)
  add_entries!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(b,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(b,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  c = create_from_nz(b)
  add_entries!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(c,nothing,PetscInt[1,1],PetscInt[1,-1])
  touch!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add!(c,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  d = create_from_nz(c)
  display(d)

  builder = SparseMatrixBuilder(Tm,MinMemory(PetscInt[1,0,0,0]))
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,PetscScalar[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(a,nothing,PetscInt[1,1],PetscInt[1,-1])
  b = nz_allocation(a)
  add_entries!(b,[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(b,nothing,PetscInt[1,1],PetscInt[1,-1])
  c = create_from_nz(b)
  add_entries!(c,[1.0 -1.0; -1.0 1.0],PetscInt[1,-1],PetscInt[-1,1])
  add_entries!(c,nothing,PetscInt[1,1],PetscInt[1,-1])
  d = create_from_nz(c)
  display(d)

  e = copy(d)
  rmul!(d,PetscScalar(2))
  @test d == 2*e
  @test nnz(d) == nnz(e)
  LinearAlgebra.fillstored!(d,zero(PetscScalar))
  @test d == zeros(PetscScalar,size(d))

  Tv = PETScVector
  builder = ArrayBuilder(Tv)
  a = nz_counter(builder,(rows,))
  @test LoopStyle(a) == DoNotLoop()
  add_entry!(a,1.0,1)
  add_entries!(a,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  add_entries!(a,nothing,PetscInt[1,1])
  b = nz_allocation(a)
  @test LoopStyle(b) == DoNotLoop()
  add_entry!(b,1.0,1)
  add_entry!(b,1.0,1)
  add_entry!(b,1.0,4)
  add_entries!(b,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  add_entries!(b,nothing,PetscInt[1,1])
  add!(b,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  touch!(b,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  c = create_from_nz(b)
  @test c === b
  add_entries!(c,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  add_entries!(c,nothing,PetscInt[1,1])
  add!(c,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  touch!(c,PetscScalar[1.0,-1.0],PetscInt[1,-1])
  d = c
  display(d)

  e = copy(d)
  rmul!(d,PetscScalar(2))
  @test d == 2*e
  fill!(d,zero(PetscScalar))
  @test d == zeros(PetscScalar,length(d))

end

end # module