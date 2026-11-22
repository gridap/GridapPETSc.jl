
"""
    get_petsc_version(handle)

Version of the PETSc library open at `handle`, as reported by the library itself.
"""
function get_petsc_version(handle)
  # These are `PetscInt*`, but we can call this before knowing the size of `PetscInt`.
  # Zeroed Int64 work for both 4 and 8 bytes (little endian).
  major, minor, patch = Ref{Int64}(0), Ref{Int64}(0), Ref{Int64}(0)
  code = ccall(
    Libdl.dlsym(handle,:PetscGetVersionNumber),
    Cint,(Ptr{Int64},Ptr{Int64},Ptr{Int64},Ptr{Cvoid}),
    major,minor,patch,C_NULL)
  iszero(code) || error("Could not determine PETSc version.")
  VersionNumber(major[],minor[],patch[])
end
