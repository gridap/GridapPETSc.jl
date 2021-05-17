module PETScArraysTests

using Test
using Gridap
using Gridap.Algebra
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt

options = "-info"

GridapPETSc.with(args=split(options)) do

  Tm = PETScMatrix
  builder = SparseMatrixBuilder(Tm)
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(a,nothing,[1,1],[1,-1])
  b = nz_allocation(a)
  add_entries!(b,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(b,nothing,[1,1],[1,-1])
  c = create_from_nz(b)
  add_entries!(c,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(c,nothing,[1,1],[1,-1])
  d = create_from_nz(c)
  display(d)

  builder = SparseMatrixBuilder(Tm,MinMemory(3))
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(a,nothing,[1,1],[1,-1])
  b = nz_allocation(a)
  add_entries!(b,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(b,nothing,[1,1],[1,-1])
  c = create_from_nz(b)
  add_entries!(c,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(c,nothing,[1,1],[1,-1])
  d = create_from_nz(c)
  display(d)

  builder = SparseMatrixBuilder(Tm,MinMemory(PetscInt[1,0,0,0]))
  rows = 1:4
  cols = 1:3
  a = nz_counter(builder,(rows,cols))
  add_entries!(a,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(a,nothing,[1,1],[1,-1])
  b = nz_allocation(a)
  add_entries!(b,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(b,nothing,[1,1],[1,-1])
  c = create_from_nz(b)
  add_entries!(c,[1.0 -1.0; -1.0 1.0],[1,-1],[-1,1])
  add_entries!(c,nothing,[1,1],[1,-1])
  d = create_from_nz(c)
  display(d)

end



end # module
