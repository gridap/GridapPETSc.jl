
struct PetscKSP
    ksp::Ref{Ptr{Cvoid}}
    function PetscKSP()
        ksp = Ref{Ptr{Cvoid}}()
        new(ksp)
    end
end

"""
    function KSPCreate!(comm::MPI.Comm, ksp::PetscKSP)

Creates the default KSP context.
"""
function KSPCreate!(comm::MPI.Comm, ksp::PetscKSP)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPCreate_ptr[],
        PetscInt,
            (MPI.Comm,
            Ptr{Ptr{Cvoid}}),
        comm, ksp.ksp)
    return error
end

"""
    function KSPCreate(comm::MPI.Comm)

Returns the default KSP context.
"""
function KSPCreate(comm::MPI.Comm)
    ksp = PetscKSP()
    error = KSPCreate!(comm, ksp)
    @assert iszero(error)
    return ksp
end

"""
    function KSPSetFromOptions!(ksp::PetscKSP)

Sets KSP options from the options database.
"""
function KSPSetFromOptions!(ksp::PetscKSP)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSetFromOptions_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},),
        ksp.ksp[])
    return error
end

"""
    function KSPSetUp!(ksp::PetscKSP)

Sets up the internal data structures for the later use of an iterative solver.
"""
function KSPSetUp!(ksp::PetscKSP)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSetUp_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},),
        ksp.ksp[])
    return error
end

"""
    function KSPSetOperators!(ksp::PetscKSP, A::PetscMat, P:: PetscMat)

Sets the matrix associated with the linear system and a (possibly) different one associated with the preconditioner.
"""
function KSPSetOperators!(ksp::PetscKSP, A::PetscMat, P:: PetscMat)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSetOperators_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid}),
        ksp.ksp[], A.mat[], P.mat[])
    return error
end

"""
    function KSPSetOperators!(ksp::PetscKSP, A::PetscMat, P:: PetscMat)

Sets the matrix associated with the linear system and a (possibly) different one associated with the preconditioner.
"""
function KSPSetTolerances!(ksp::PetscKSP, rtol::AbstractFloat, abstol::AbstractFloat, dtol::AbstractFloat, maxits::Integer)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSetTolerances_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            PetscReal,
            PetscReal,
            PetscReal,
            PetscInt),
        ksp.ksp[], rtol, abstol, dtol, maxits)
    return error
end

"""
    function KSPSolve!(ksp::PetscKSP, b::PetscVec, x::PetscVec)

Solves linear system.
"""
function KSPSolve!(ksp::PetscKSP, b::PetscVec, x::PetscVec)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSolve_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid}),
        ksp.ksp[], b.vec[], x.vec[])
    return error
end

"""
    function KSPSolveTranspose!(arg1::Ptr{Cvoid}, arg2::AbstractArray, arg3::AbstractArray)

Solves the transpose of a linear system.
"""
function KSPSolveTranspose!(arg1::Ptr{Cvoid}, arg2::AbstractArray, arg3::AbstractArray)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPSolveTranspose_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            AbstractArray,
            AbstractArray),
        arg1, arg2, arg3)
    return error
end

"""
    function KSPDestroy!(ksp::PetscKSP)

Destroys KSP context.
"""
function KSPDestroy!(ksp::PetscKSP)
    @check_if_loaded
    @check_if_initialized
    error = ccall( KSPDestroy_ptr[],
        PetscErrorCode,
            (Ptr{Ptr{Cvoid}},),
        ksp.ksp)
    return error
end

"""
    function KSPGetIterationNumber!(ksp::PetscKSP, its::Integer)

Gets the current iteration number; if the KSPSolve() is complete, returns the number of iterations used.
"""
function KSPGetIterationNumber!(ksp::PetscKSP)
    @check_if_loaded
    @check_if_initialized
    its = Vector{PetscInt}(undef,1)
    error = ccall( KSPGetIterationNumber_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},Ptr{PetscInt}),
        ksp.ksp[], its )
    @assert iszero(error)
    return its[1]
end
