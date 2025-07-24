
using GridapPETSc
using PartitionedArrays


parts = with_mpi() do distribute
  distribute(LinearIndices((4,)))
end

I1 = Base.OneTo(20)
I2 = [1,4,2,5,3,6,10,7,8]
I3 = PRange(uniform_partition(parts,4,20,true,false))

GridapPETSc.with() do
  J1 = PETScIndexSet(I1)
  J2 = PETScIndexSet(I2)
  J3 = PETScIndexSet(I3)
end
