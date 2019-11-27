
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


function PetscInitializeNoArguments!()
    @check_if_loaded
    error = ccall( PetscInitializeNoArguments_ptr[], Int, ())
    return error
end


function PetscFinalize!()
    @check_if_loaded
    error = ccall( PetscFinalize_ptr[], Int, ())
    return error
end


function init!()
    if (PetscInitialized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end

    error = PetscInitializeNoArguments!()
    @assert iszero(error)
end


function init!(args)
    init!(args, "", "")
end


function init!(args::Vector{String}, filename::String, help::String)
    args = ["julia";args];

    if (PetscInitialized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end

    error = PetscInitializeNoPointers!(args,filename,help);
    @assert iszero(error)
end


function finalize!()
    if (!PetscFinalized()) 
        error = PetscFinalize!() 
        @assert iszero(error)
    end
end



