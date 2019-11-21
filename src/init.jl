
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


function PetscInitializeNoPointers(args::Vector{String}, filename::String, help::String)
    nargs = Cint(length(args))
    error = ccall(
        (:PetscInitializeNoPointers, PETSC_LIB), 
            PetscErrorCode, 
                (Cint, 
                Ptr{Ptr{UInt8}}, 
                Cstring, 
                Cstring), 
            nargs, args, filename, help)
    @assert iszero(error)
end


function PetscInitializeNoArguments()
    @check_if_loaded
    error = ccall( PetscInitializeNoArguments_ptr[], Int, ())
    @assert iszero(error)
end


function PetscInitialize()

    if (PetscInitialized()) 
        PetscFinalize() 
    end

    PetscInitializeNoArguments()
end


function PetscInitialize(args)
    PetscInitialize(args, "", "")
end


function PetscInitialize(args::Vector{String}, filename::String, help::String)
    args = ["julia";args];

    if (PetscInitialized()) 
        PetscFinalize() 
    end

    PetscInitializeNoPointers(args,filename,help);
end


function PetscFinalize()
    @check_if_loaded
    error = ccall( PetscFinalize_ptr[], Int, ())
    @assert iszero(error)
end
