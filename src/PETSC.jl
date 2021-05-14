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

macro wrapper(fn,rt,argts,args,url)
  sargs = "$(args)"
  if length(args.args) == 1
    sargs = sargs[1:end-2]*")"
  end
  if isempty(rstrip(url))
    str = """
        $(fn.value)$(sargs)
    """
  else
    str = """
        $(fn.value)$(sargs)

    See [PETSc manual]($url).
    """
  end
  expr = quote
    @doc $str
    function $(fn.value)($(args.args...))
      ccall(
        Libdl.dlsym(libpetsc_handle[],$fn),
        $rt,$argts,$(args.args...))
    end
  end
  esc(expr)
end

#Petsc init related functions

@wrapper(:PetscInitializeNoArguments,PetscErrorCode,(),(),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatEqual.html")
@wrapper(:PetscInitializeNoPointers,PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),(argc,args,filename,help),"")
@wrapper(:PetscFinalize,PetscErrorCode,(),(),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalize.html")
@wrapper(:PetscFinalized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalized.html")
@wrapper(:PetscInitialized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitialized.html")

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

@wrapper(:PETSC_VIEWER_STDOUT_,PetscViewer,(MPI.Comm,),(comm,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_.html")
@wrapper(:PETSC_VIEWER_DRAW_,PetscViewer,(MPI.Comm,),(comm,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_.html")

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

@wrapper(:VecCreateSeqWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,array,vec),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeqWithArray.html")
@wrapper(:VecDestroy,PetscErrorCode,(Ptr{Vec},),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDestroy.html")
@wrapper(:VecView,PetscErrorCode,(Vec,PetscViewer),(vec,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecView.html")
@wrapper(:VecSetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(x,ni,ix,y,iora),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetValues.html")
@wrapper(:VecGetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(x,ni,ix,y),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetValues.html")
@wrapper(:VecAssemblyBegin,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyBegin.html")
@wrapper(:VecAssemblyEnd,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyEnd.html")

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

@wrapper(:MatCreateSeqAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,nz,nnz,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJ.html")
@wrapper(:MatCreateSeqAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,i,j,a,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJWithArrays.html")
@wrapper(:MatDestroy,PetscErrorCode,(Ptr{Mat},),(A,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDestroy.html")
@wrapper(:MatView,PetscErrorCode,(Mat,PetscViewer),(mat,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatView.html")
@wrapper(:MatSetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(mat,m,idxm,n,idxn,v,addv),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValues.html")
@wrapper(:MatGetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(mat,m,idxm,n,idxn,v),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetValues.html")
@wrapper(:MatAssemblyBegin,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyBegin.html")
@wrapper(:MatAssemblyEnd,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyEnd.html")
@wrapper(:MatGetSize,PetscErrorCode,(Mat,Ptr{PetscInt},Ptr{PetscInt}),(mat,m,n),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetSize.html")
@wrapper(:MatEqual,PetscErrorCode,(Mat,Mat,Ptr{PetscBool}),(A,B,flg),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatEqual.html")

end # module
