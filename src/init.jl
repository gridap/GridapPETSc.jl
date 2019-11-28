macro check_if_mpi_initialized()
  quote
    if ! MPI.Initialized()
        error("MPI is not initialized. Please use MPI.Init() method.")
    end
  end
end

macro check_if_initialized()
  quote
    if ! PetscInitialized()
      error("PETSc is not initialized. Please use GridapPETSc.Init!() method.")
    end
  end
end

"""
    function PetscInitialized()

Determine whether PETSc is initialized. 
"""
function PetscInitialized()
    init = Array{PetscBool}(undef,1);
    error = ccall( 
        (:PetscInitialized,  PETSC_LIB),
            PetscErrorCode, 
                (Ptr{PetscBool},), 
            init);
    @assert iszero(error)
    return init[1] == PETSC_TRUE
end

"""
    function PetscFinalized()

Determine whether PetscFinalize() has been called yet 
"""
function PetscFinalized()
    init = Array{PetscBool}(undef,1);
    error = ccall( 
        (:PetscFinalized,  PETSC_LIB),
            PetscErrorCode, 
                (Ptr{PetscBool},), 
            init);
    @assert iszero(error)
    return init[1] == PETSC_TRUE
end

"""
    function PetscInitializeNoPointers!(args::Vector{String}, filename::String, help::String)

Calls PetscInitialize() without the pointers to argc and args.
This is called only by the PETSc Julia interface. 
Even though it might start MPI it sets the flag to
indicate that it did NOT start MPI so that the PetscFinalize() 
does not end MPI, thus allowing PetscInitialize() to
be called multiple times from Julia without the problem 
of trying to initialize MPI more than once.
"""
function PetscInitializeNoPointers!(args::Vector{String}, filename::String, help::String)
    nargs = Cint(length(args))
    error = ccall(
        (:PetscInitializeNoPointers, PETSC_LIB), 
            PetscErrorCode, 
                (Cint, 
                Ptr{Ptr{UInt8}}, 
                Cstring, 
                Cstring), 
            nargs, args, filename, help)
    return error
end

"""
    function PetscInitializeNoArguments!()

Calls PetscInitialize() without the command line arguments. 
"""
function PetscInitializeNoArguments!()
    @check_if_loaded
    error = ccall( PetscInitializeNoArguments_ptr[], Int, ())
    return error
end

"""
    function PetscFinalize!()

Checks for options to be called at the conclusion of the program. 
MPI_Finalize() is called only if the user had not called MPI_Init() 
before calling PetscInitialize(). 
"""
function PetscFinalize!()
    @check_if_loaded
    error = ccall( PetscFinalize_ptr[], Int, ())
    return error
end

"""
    function Init!()

Initialize Petsc library.
"""
function Init!()
    @check_if_mpi_initialized
    if (PetscInitialized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end
    error = PetscInitializeNoArguments!()
    @assert iszero(error)
end

"""
    function Init!(args)

Initialize Petsc library.
"""
function Init!(args)
    Init!(args, "", "")
end


"""
    function Init!(args::Vector{String}, filename::String, help::String)

Initialize Petsc library.
"""
function Init!(args::Vector{String}, filename::String, help::String)
    @check_if_mpi_initialized
    if (PetscInitialized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end
    args = ["julia";args];
    error = PetscInitializeNoPointers!(args,filename,help);
    @assert iszero(error)
end


"""
    function Finalize!()

Finalize Petsc library.
"""
function Finalize!()
    if (!PetscFinalized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end
end

