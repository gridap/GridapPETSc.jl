module PETScArraysTests

using GridapPETSc
using Test
using SparseArrays
using SparseMatricesCSR
using GridapPETSc: PetscScalar, PetscInt
using LinearAlgebra

options = "-info"
GridapPETSc.with(args=split(options)) do

  n = 10
  v = PETScVector(n)

  @test length(v) == n
  v[4] = 30
  @test 30 == v[4]

  s = similar(v,PetscScalar,4)
  w = similar(v)

  a = zeros(PetscScalar,10)
  b = convert(Vector{PetscScalar},a)
  @test a === b

  a = zeros(PetscScalar,10)
  b = convert(PETScVector,a)
  @test a === b.ownership

  a = zeros(PetscInt,10)
  b = convert(Vector{PetscScalar},a)
  @test a !== b

  a = PETScVector(10)
  b = convert(PETScVector,a)
  @test a === b

  m = 4
  n = 5
  A = PETScMatrix(m,n)
  @test size(A) == (m,n)
  A[1,3] = 5
  A[3,5] = 7
  @test A[1,3] == 5
  @test A[3,5] == 7
  display(A)

  B = similar(A,PetscScalar,3,2)
  @test typeof(A) == typeof(B)
  @test size(B) == (3,2)

  I = PetscInt[1,1,2,2,2,3,3,3,4,4]
  J = PetscInt[1,2,1,2,3,2,3,4,3,4]
  V = PetscScalar[4,-2,-1,6,-2,-1,6,-2,-1,4]
  m = PetscInt(4)
  n = PetscInt(4)

  Ccsr = sparsecsr(I,J,V,m,n)
  C = petsc_sparse(I,J,V,m,n)
  @test nnz(C) == nnz(Ccsr)

  x = similar(v,size(C,2))
  fill!(x,1)
  y = C*x
  @test typeof(y) == typeof(x)
  @test y == [2,3,3,3]

  y = 2*x
  @test typeof(y) == typeof(x)
  y = x*2
  @test typeof(y) == typeof(x)

  D = 2*C
  @test typeof(D) == typeof(C)
  D = C*3
  @test typeof(D) == typeof(C)

  n = 4
  aj = rand(PetscScalar,n)
  ap = convert(PETScVector,aj)
  @test ap == aj
  @test norm(ap) ≈ norm(aj)
  @test ap+2*ap == ap+2*ap
  @test typeof(ap+2*ap) ==  PETScVector
  @test ap-2*ap == ap-2*ap
  @test typeof(ap-2*ap) ==  PETScVector

  @test typeof(C*ap) == PETScVector
  @test typeof(C*aj) == PETScVector
  @test C*ap == C*aj

end

end # module
