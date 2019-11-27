
struct PetscMat 
    mat::Ref{Ptr{Cvoid}}
    function PetscMat()
        mat = Ref{Ptr{Cvoid}}()
        new(mat)
    end
end

function  MatCreateSeqAIJWithArrays!(
        comm::MPI.Comm,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{Float64},
        mat::PetscMat)
    @check_if_loaded
    error = ccall( MatCreateSeqAIJWithArrays_ptr[],
            PetscErrorCode,
                (MPI.Comm,
                PetscInt,
                PetscInt,
                Ptr{PetscInt},
                Ptr{PetscInt},
                Ptr{PetscScalar},
                Ptr{Cvoid}),
            comm, m, n, i, j, a, mat.mat)
    return error
end

function MatDestroy!(mat::PetscMat)
    error = ccall( 
            (:MatDestroy, PETSC_LIB), 
            PetscErrorCode, 
                (Ptr{Cvoid},), 
            mat.mat)
    return error
end


function MatView(mat::PetscMat, viewer::PetscViewer=C_NULL)
    error = ccall( 
            (:MatView,  PETSC_LIB), 
            PetscErrorCode, 
                (Ptr{Cvoid}, 
                Ptr{Cvoid}), 
            mat.mat[], viewer);
    return error
end

