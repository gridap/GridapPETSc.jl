"""
Low level interface with PETSC, which serve as the back-end in GridapPETSc.

The types and functions defined here are almost 1-to-1 to the corresponding C counterparts. In particular, the types defined
can be directly used to call C PETSc routines via `ccall`.
When a C function expects a pointer, use a `Ref` to the corresponding Julia alias.
E.g., if an argument is `PetscBool *` in the C code, pass an object with type `Ref{PetscBool}` from the Julia code.
Using this rule, [`PETSC.PetscInitialized`](@ref) can be called as

    flag = Ref{PetscBool}()
    @check_error_code PetscInitialized(flag)
    if flag[] == PETSC_TRUE
      println("Petsc is initialized!")
    end
"""
module PETSC

using Libdl
using GridapPETSc: libpetsc_handle
using MPI

include("Config.jl")

let types_jl = joinpath(@__DIR__,"..","deps","PetscDataTypes.jl")
  if !isfile(types_jl)
    msg = """
    GridapPETSc needs to be configured before use. Type
  
    pkg> build
  
    and try again.
    """
    error(msg)
  end
  
  include(types_jl)
end

#Petsc init related functions

"""
    PetscInitializeNoArguments()

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitializeNoArguments.html).
"""
function PetscInitializeNoArguments()
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscInitializeNoArguments),
    PetscErrorCode,())
end

"""
    PetscInitializeNoPointers(argc,args,filename,help)
"""
function PetscInitializeNoPointers(argc,args,filename,help)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscInitializeNoPointers),
    PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),
    argc,args,filename,help)
end

"""
    PetscFinalize()

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalize.html).
"""
function PetscFinalize()
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscFinalize),PetscErrorCode,())
end

"""
    PetscFinalized(flag)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalized.html).
"""
function PetscFinalized(flag)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscFinalized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

"""
    PetscInitialized(flag)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitialized.html).
"""
function PetscInitialized(flag)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscInitialized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

# viewer related functions

"""
Julia alias for `PetscViewer` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PetscViewer.html).
"""
struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)

"""
    PETSC_VIEWER_STDOUT_(comm)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_.html).
"""
function PETSC_VIEWER_STDOUT_(comm)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PETSC_VIEWER_STDOUT_),
    PetscViewer,(MPI.Comm,),comm)
end

"""
    @PETSC_VIEWER_STDOUT_SELF

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_SELF.html).
"""
macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end 

"""
    @PETSC_VIEWER_STDOUT_WORLD

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_WORLD.html).
"""
macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end 

"""
    PETSC_VIEWER_DRAW_(comm)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_.html).
"""
function PETSC_VIEWER_DRAW_(comm)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PETSC_VIEWER_DRAW_),
    PetscViewer,(MPI.Comm,),comm)
end

"""
    @PETSC_VIEWER_DRAW_SELF

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_SELF.html).
"""
macro PETSC_VIEWER_DRAW_SELF()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_SELF)
  end
end 

"""
    @PETSC_VIEWER_DRAW_WORLD

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_WORLD.html).
"""
macro PETSC_VIEWER_DRAW_WORLD()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_WORLD)
  end
end 

# Vector related functions

"""
Julia alias for the `InsertMode` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/InsertMode.html).
"""
@enum InsertMode begin
  NOT_SET_VALUES
  INSERT_VALUES
  ADD_VALUES
  MAX_VALUES
  MIN_VALUES
  INSERT_ALL_VALUES
  ADD_ALL_VALUES
  INSERT_BC_VALUES
  ADD_BC_VALUES
end

"""
Julia alias for the `Vec` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/Vec.html).
"""
struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)

"""
    VecCreateSeqWithArray(comm,bs,n,array,vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeqWithArray.html).
"""
function VecCreateSeqWithArray(comm,bs,n,array,vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecCreateSeqWithArray),
    PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),
    comm,bs,n,array,vec)
end

"""
    VecDestroy(vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDestroy.html).
"""
function VecDestroy(vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecDestroy),
    PetscErrorCode,(Ptr{Vec},),
    vec)
end

"""
    VecView(vec,viewer)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecView.html).
"""
function VecView(vec,viewer)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecView),
    PetscErrorCode,(Vec,PetscViewer),vec,viewer)
end

"""
    VecSetValues(x,ni,ix,y,iora)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetValues.html).
"""
function VecSetValues(x,ni,ix,y,iora)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecSetValues),
    PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),
    x,ni,ix,y,iora)
end

"""
    VecGetValues(x,ni,ix,y)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetValues.html).
"""
function VecGetValues(x,ni,ix,y)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecGetValues),
    PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),
    x,ni,ix,y)
end

"""
    VecAssemblyBegin(vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyBegin.html).
"""
function VecAssemblyBegin(vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecAssemblyBegin),
    PetscErrorCode,(Vec,), vec)
end

"""
    VecAssemblyEnd(vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyEnd.html).
"""
function VecAssemblyEnd(vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecAssemblyEnd),
    PetscErrorCode,(Vec,), vec)
end

# Matrix related functions


"""
Julia alias for the `MatAssemblyType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyType.html#MatAssemblyType).
"""
@enum MatAssemblyType begin
  MAT_FINAL_ASSEMBLY=0
  MAT_FLUSH_ASSEMBLY=1
end

"""
Julia constant storing the `PETSC_DEFAULT` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DEFAULT.html).
"""
const PETSC_DEFAULT = Cint(-2)

"""
Julia constant storing the `PETSC_DECIDE` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DECIDE.html).
"""
const PETSC_DECIDE = Cint(-1)

"""
Julia constant storing the `PETSC_DETERMINE` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DETERMINE.html).
"""
const PETSC_DETERMINE = PETSC_DECIDE

"""
Julia alias for the `Mat` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/Mat.html).
"""
struct Mat
  ptr::Ptr{Cvoid}
end
Mat() = Mat(Ptr{Cvoid}())
Base.convert(::Type{Mat},p::Ptr{Cvoid}) = Mat(p)

"""
    MatCreateSeqAIJ(comm,m,n,nz,nnz,mat)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJ.html).
"""
function MatCreateSeqAIJ(comm,m,n,nz,nnz,mat)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatCreateSeqAIJ),
    PetscErrorCode,
    (MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),
    comm,m,n,nz,nnz,mat)
end

"""
    MatCreateSeqAIJWithArrays(comm,m,n,i,j,a,mat)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJWithArrays.html).
"""
function MatCreateSeqAIJWithArrays(comm,m,n,i,j,a,mat)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatCreateSeqAIJWithArrays),
    PetscErrorCode,
    (MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),
    comm,m,n,i,j,a,mat)
end

"""
    MatDestroy(A)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDestroy.html).
"""
function MatDestroy(A)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatDestroy),
    PetscErrorCode,(Ptr{Mat},),
    A)
end

"""
    MatView(mat,viewer)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatView.html).
"""
function MatView(mat,viewer)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatView),
    PetscErrorCode,(Mat,PetscViewer),mat,viewer)
end

"""
    MatSetValues(mat,m,idxm,n,idxn,v,addv)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValues.html).
"""
function MatSetValues(mat,m,idxm,n,idxn,v,addv)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatSetValues),
    PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),
    mat,m,idxm,n,idxn,v,addv)
end

"""
    MatGetValues(mat,m,idxm,n,idxn,v)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetValues.html).
"""
function MatGetValues(mat,m,idxm,n,idxn,v)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatGetValues),
    PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),
    mat,m,idxm,n,idxn,v)
end

"""
    MatAssemblyBegin(mat,typ)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyBegin.html).
"""
function MatAssemblyBegin(mat,typ)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatAssemblyBegin),
    PetscErrorCode,(Mat,MatAssemblyType), mat, typ)
end

"""
    MatAssemblyEnd(mat,typ)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyEnd.html).
"""
function MatAssemblyEnd(mat,typ)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:MatAssemblyEnd),
    PetscErrorCode,(Mat,MatAssemblyType), mat, typ)
end

end # module
