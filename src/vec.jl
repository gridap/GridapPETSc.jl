
struct PetscVec 
    vec::Ref{Ptr{Cvoid}}
    function PetscVec()
        vec = Ref{Ptr{Cvoid}}()
        new(vec)
    end
end

"""
    function VecCreateSeqWithArray!(
        comm::MPI.Comm,
        bs::Int,
        n::Int,
        array::Vector{PetscScalar},
        vec::PetscVec)

Creates a standard, sequential array-style vector, 
where the user provides the array space to store the vector values. 
"""
function VecCreateSeqWithArray!(
        comm::MPI.Comm,
        bs::Int,
        n::Int,
        array::Vector{PetscScalar},
        vec::PetscVec)
    @check_if_loaded
    @check_if_initialized
    error = ccall( VecCreateSeqWithArray_ptr[],
            PetscErrorCode,
                (MPI.Comm,
                PetscInt,
                PetscInt,
                Ptr{PetscScalar},
                Ptr{Cvoid}),
            comm, bs, n, array, vec.vec)
    return error
end

"""
    function VecCreateSeqWithArray(
        comm::MPI.Comm,
        bs::Int,
        n::Int,
        array::Vector{PetscScalar})

Returns a standard, sequential array-style vector, 
where the user provides the array space to store the vector values. 
"""
function VecCreateSeqWithArray(
        comm::MPI.Comm,
        bs::Int,
        n::Int,
        array::Vector{PetscScalar})
    Vec = PetscVec()
    error = VecCreateSeqWithArray!(comm, bs, n, array, Vec)
    @assert iszero(error)
    return Vec
end

"""
    function VecDestroy!(vec::PetscVec)

Destroys a vector. 
"""
function VecDestroy!(vec::PetscVec)
    @check_if_loaded
    @check_if_initialized
    error = ccall( VecDestroy_ptr[],
            PetscErrorCode, 
                (Ptr{Cvoid},), 
            vec.vec)
    return error
end

"""
    function VecView(vec::PetscVec, viewer::PetscViewer=C_NULL)

Views a vector object. 
"""
function VecView(vec::PetscVec, viewer::PetscViewer=C_NULL)
    @check_if_loaded
    @check_if_initialized
    error = ccall( VecView_ptr[],
            PetscErrorCode, 
                (Ptr{Cvoid},
                Int64),
            vec.vec[],viewer);
    return error
end

