
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

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalize.html)
"""
function PetscFinalize()
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscFinalize),PetscErrorCode,())
end

"""
    PetscFinalized(flag)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalized.html)
"""
function PetscFinalized(flag)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscFinalized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

"""
    PetscInitialized(flag)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitialized.html)
"""
function PetscInitialized(flag)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscInitialized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

# viewer related functions

"""
Julia alias for `PetscViewer` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PetscViewer.html)
"""
struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)

"""
    PETSC_VIEWER_STDOUT_(comm)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_.html)
"""
function PETSC_VIEWER_STDOUT_(comm)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PETSC_VIEWER_STDOUT_),
    PetscViewer,(MPI.Comm,),comm)
end

"""
    @PETSC_VIEWER_STDOUT_SELF

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_SELF.html)
"""
macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end 

"""
    @PETSC_VIEWER_STDOUT_WORLD

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_WORLD.html)
"""
macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end 

# Vector related functions

"""
Julia alias for the `InsertMode` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/InsertMode.html)
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

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/Vec.html)
"""
struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)

"""
    VecCreateSeqWithArray(comm,bs,n,array,vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeqWithArray.html)
"""
function VecCreateSeqWithArray(comm,bs,n,array,vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecCreateSeqWithArray),
    PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),
    comm,bs,n,array,vec)
end

"""
    VecDestroy(vec)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDestroy.html)
"""
function VecDestroy(vec)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecDestroy),
    PetscErrorCode,(Ptr{Vec},),
    vec)
end

"""
    VecView(vec,viewer)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecView.html)
"""
function VecView(vec,viewer)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecView),
    PetscErrorCode,(Vec,PetscViewer),vec,viewer)
end

"""
    VecSetValues(x,ni,ix,y,iora)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetValues.html#VecSetValues)
"""
function VecSetValues(x,ni,ix,y,iora)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:VecSetValues),
    PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),
    x,ni,ix,y,iora)
end

