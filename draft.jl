module TMP

using Libdl
using MPI

# Linking

PETSC_LIB = "/usr/lib/petsc/lib/libpetsc_real.so"
flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
const PETSC = Libdl.dlopen(PETSC_LIB, flags)

# Petsc types not dependent on configuration

const PetscErrorCode = Cint

@enum PetscBool PETSC_FALSE PETSC_TRUE

@enum PetscDataType begin
  PETSC_DATATYPE_UNKNOWN = 0
  PETSC_DOUBLE = 1
  PETSC_COMPLEX = 2
  PETSC_LONG = 3
  PETSC_SHORT = 4
  PETSC_FLOAT = 5
  PETSC_CHAR = 6
  PETSC_BIT_LOGICAL = 7
  PETSC_ENUM = 8
  PETSC_BOOL = 9
  PETSC___FLOAT128 = 10
  PETSC_OBJECT = 11
  PETSC_FUNCTION = 12
  PETSC_STRING = 13
  PETSC___FP16 = 14
  PETSC_STRUCT = 15
  PETSC_INT = 16
  PETSC_INT64 = 17
end

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

# Handling of error codes

struct PetscError <: Exception
  code::PetscErrorCode
end
Base.showerror(io::IO, e::PetscError) = print(io, "Petsc returned with error code: $(Int(e.code)) ")

macro petscerror(expr)
  quote
    code = $(esc(expr))
    if code != zero(PetscErrorCode)
      throw(PetscError(code))
    end
  end
end

# Petsc datatype related functions

function PetscDataTypeFromString(name,ptype,found)
  ccall(
    Libdl.dlsym(PETSC,:PetscDataTypeFromString),
    PetscErrorCode,(Cstring,Ptr{PetscDataType},Ptr{PetscBool}),
    name,ptype,found)
end

function PetscDataTypeGetSize(ptype,size)
  ccall(
    Libdl.dlsym(PETSC,:PetscDataTypeGetSize),
    PetscErrorCode,(PetscDataType,Ptr{Csize_t}),
    ptype,size)
end

# Data types depending on configuration

real_type = Ref{PetscDataType}()
scalar_type = Ref{PetscDataType}()
int_type = Ref{PetscDataType}()
real_found = Ref{PetscBool}()
scalar_found = Ref{PetscBool}()
int_found = Ref{PetscBool}()
real_size = Ref{Csize_t}()
scalar_size = Ref{Csize_t}()
int_size = Ref{Csize_t}()
@petscerror PetscDataTypeFromString("Real",real_type,real_found)
@petscerror PetscDataTypeFromString("Scalar",scalar_type,scalar_found)
@petscerror PetscDataTypeFromString("Int",int_type,int_found)
@assert real_found[] == PETSC_TRUE
@assert scalar_found[] == PETSC_TRUE
@assert int_found[] == PETSC_TRUE
@petscerror PetscDataTypeGetSize(real_type[],real_size)
@petscerror PetscDataTypeGetSize(scalar_type[],scalar_size)
@petscerror PetscDataTypeGetSize(int_type[],int_size)

if real_type[] == PETSC_DOUBLE &&  real_size[] == 8
  const PetscReal = Float64
elseif real_type[] == PETSC_DOUBLE &&  real_size[] == 4
  const PetscReal = Float32
else
  throw("Cannot determine PetscReal type")
end

if scalar_type[] == PETSC_DOUBLE &&  scalar_size[] == 8
  const PetscScalar = Float64
elseif scalar_type[] == PETSC_DOUBLE &&  scalar_size[] == 4
  const PetscScalar = Float32
else
  throw("Cannot determine PetscScalar type")
end

if int_type[] in (PETSC_INT, PETSC_DATATYPE_UNKNOWN) &&  int_size[] == 8
  const PetscInt = Int64
elseif int_type[] in (PETSC_INT, PETSC_DATATYPE_UNKNOWN) &&  int_size[] == 4
  const PetscInt = Int32
else
  throw("Cannot determine PetscInt type")
end

#Petsc init related functions

function PetscInitializeNoArguments()
  ccall(
    Libdl.dlsym(PETSC,:PetscInitializeNoArguments),
    PetscErrorCode,())
end

function PetscInitializeNoPointers(argc,args,filename,help)
  ccall(
    Libdl.dlsym(PETSC,:PetscInitializeNoPointers),
    PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),
    argc,args,filename,help)
end

function PetscFinalize()
  ccall(
    Libdl.dlsym(PETSC,:PetscFinalize),PetscErrorCode,())
end

function PetscFinalized(flag)
  ccall(
    Libdl.dlsym(PETSC,:PetscFinalized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

function PetscInitialized(flag)
  ccall(
    Libdl.dlsym(PETSC,:PetscInitialized),
    PetscErrorCode,(Ptr{PetscBool},),flag)
end

# Automatically call PetscFinalize
# Important if the computation is aborted by julia

atexit() do
  msg = "Unable to properly finalize Petsc in the `atexit` hook."
  flag = Ref{PetscBool}()
  code = PetscInitialized(flag)
  if code != zero(PetscErrorCode)
    println(msg)
    return nothing
  end
  if flag[]==PETSC_TRUE
    code = PetscFinalize()
    if code != zero(PetscErrorCode)
      println(msg)
      return nothing
    end
  end
  nothing
end

# viewer related functions

struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)

function PETSC_VIEWER_STDOUT_(comm)
  ccall(
    Libdl.dlsym(PETSC,:PETSC_VIEWER_STDOUT_),
    PetscViewer,(MPI.Comm,),comm)
end

macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end 

macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end 

# Vector related functions

struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)

function VecCreateSeqWithArray(comm,bs,n,array,vec)
  ccall(
    Libdl.dlsym(PETSC,:VecCreateSeqWithArray),
    PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),
    comm,bs,n,array,vec)
end

function VecDestroy(vec)
  ccall(
    Libdl.dlsym(PETSC,:VecDestroy),
    PetscErrorCode,(Ptr{Vec},),
    vec)
end

function VecView(vec,viewer)
  ccall(
    Libdl.dlsym(PETSC,:VecView),
    PetscErrorCode,(Vec,PetscViewer),vec,viewer)
end

function VecSetValues(x,ni,ix,y,iora)
  ccall(
    Libdl.dlsym(PETSC,:VecSetValues),
    PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),
    x,ni,ix,y,iora)
end

# Demo

if !MPI.Initialized()
    MPI.Init()
end

flag = Ref{PetscBool}()
@petscerror PetscInitialized(flag)
if flag[] == PETSC_TRUE
  @petscerror PetscFinalize()
end
args = ["julia"]
argc = length(args)
file = ""
help = ""
@petscerror PetscInitializeNoPointers(argc,args,file,help)

comm = MPI.COMM_SELF
bs = PetscInt(1)
array = PetscScalar[1,2,3,4,1]
n = PetscInt(length(array))
vec = Ref{Vec}()
@petscerror VecCreateSeqWithArray(comm,bs,n,array,vec)

@petscerror VecView(vec[],@PETSC_VIEWER_STDOUT_SELF)

ids = PetscInt[0,1,3]
vals = PetscScalar[10,20,30]
@petscerror VecSetValues(vec[],length(ids),ids,vals,INSERT_VALUES)

@petscerror VecView(vec[],@PETSC_VIEWER_STDOUT_WORLD)
@petscerror VecView(vec[],C_NULL)

@petscerror VecDestroy(vec)

@petscerror PetscFinalize()

end # module
