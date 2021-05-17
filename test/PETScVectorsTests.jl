module PETScVectorsTests

using GridapPETSc
using Test
using GridapPETSc: PetscScalar

options = "-info"
GridapPETSc.with(args=split(options)) do 

  n = 10
  v = PETScVector(n)
  
  @test length(v) == n
  v[4] = 30
  @test 30 == v[4]
  
  s = similar(v,PetscScalar,4)
  w = similar(v)
  
  m = 4
  n = 5
  A = PETScMatrix(m,n)
  @test size(A) == (m,n)
  A[1,3] = 5
  A[3,5] = 7
  @test A[1,3] == 5
  @test A[3,5] == 7
  
  B = similar(A,PetscScalar,3,2)
  @test typeof(A) == typeof(B)
  @test size(B) == (3,2)

end

end # module
