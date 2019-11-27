
struct PetscKSP
    ksp::Ref{Ptr{Cvoid}}
    function PetscKSP()
        ksp = Ref{Ptr{Cvoid}}()
        new(ksp)
    end
end

function KSPCreate!(comm::MPI.Comm, ksp::PetscKSP)
    @check_if_loaded
    error = ccall( KSPCreate_ptr[],
        PetscInt,
            (MPI.Comm,
            Ptr{Ptr{Cvoid}}),
        comm, ksp.ksp)
    return error
end

function KSPCreate(comm::MPI.Comm)
    ksp = PetscKSP()
    error = KSPCreate!(comm, ksp)
    @assert iszero(error)
    return ksp
end

function KSPSetOperators!(ksp::PetscKSP, A::PetscMat, P:: PetscMat)
    @check_if_loaded
    error = ccall( KSPSetOperators_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid}),
        ksp.ksp[], A.mat[], P.mat[])
    return error
end


function KSPSolve!(ksp::PetscKSP, b::PetscVec, x::PetscVec)
    @check_if_loaded
    error = ccall( KSPSolve_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid}),
        ksp.ksp[], b.vec[], x.vec[])
    return error
end


function KSPSolveTranspose!(arg1::Ptr{Cvoid}, arg2::AbstractArray, arg3::AbstractArray)
    @check_if_loaded
    error = ccall( KSPSolveTranspose_ptr[],
        PetscErrorCode,
            (Ptr{Cvoid},
            AbstractArray,
            AbstractArray),
        arg1, arg2, arg3)
    return error
end

function KSPDestroy!(ksp::PetscKSP)
    @check_if_loaded
    error = ccall( KSPDestroy_ptr[],
        PetscErrorCode,
            (Ptr{Ptr{Cvoid}},),
        ksp.ksp)
    return error
end


