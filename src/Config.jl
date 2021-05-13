
"""
Julia alias to `PetscErrorCode` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscErrorCode.html).
"""
const PetscErrorCode = Cint

"""
Julia alias to `PetscBool` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscBool.html).
"""
@enum PetscBool PETSC_FALSE PETSC_TRUE

"""
Julia alias to `PetscDataType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscDataType.html).
"""
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

"""
    PetscDataTypeFromString(name,ptype,found)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscDataTypeFromString.html).
"""
function PetscDataTypeFromString(name,ptype,found)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscDataTypeFromString),
    PetscErrorCode,(Cstring,Ptr{PetscDataType},Ptr{PetscBool}),
    name,ptype,found)
end

"""
    PetscDataTypeGetSize(ptype,size)

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscDataTypeGetSize.html).
"""
function PetscDataTypeGetSize(ptype,size)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscDataTypeGetSize),
    PetscErrorCode,(PetscDataType,Ptr{Csize_t}),
    ptype,size)
end

"""
    struct PetscError <: Exception
      code::PetscErrorCode
    end

Custom `Exception` thrown by [`@check_error_code`](@ref).
"""
struct PetscError <: Exception
  code::PetscErrorCode
end
Base.showerror(io::IO, e::PetscError) = print(io, "Petsc returned with error code: $(Int(e.code)) ")

"""
    @check_error_code expr

Check if `expr` returns an error code equal to `zero(PetscErrorCode)`.
If not, throw an instance of [`PetscError`](@ref).
"""
macro check_error_code(expr)
  quote
    code = $(esc(expr))
    if code != zero(PetscErrorCode)
      throw(PetscError(code))
    end
  end
end

