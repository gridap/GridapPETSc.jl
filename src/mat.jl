
struct PetscMat 
    mat::Ref{Ptr{Cvoid}}
    function PetscMat()
        mat = Ref{Ptr{Cvoid}}()
        new(mat)
    end
end

"""
    function  MatCreateSeqBAIJWithArrays!(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)

Creates a sequential block AIJ matrix using matrix elements 
(in CSR format) provided by the user. 
"""
function  MatCreateSeqBAIJWithArrays!(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)
    @check_if_loaded
    @check_if_initialized
    error = ccall( MatCreateSeqBAIJWithArrays_ptr[],
            PetscErrorCode,
                (MPI.Comm,
                PetscInt,
                PetscInt,
                PetscInt,
                Ptr{PetscInt},
                Ptr{PetscInt},
                Ptr{PetscScalar},
                Ptr{Cvoid}),
            comm, bs, m, n, i, j, a, mat.mat)
    return error
end

"""
    function  MatCreateSeqBAIJWithArrays(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})

Returns a sequential block AIJ matrix using matrix elements 
(in CSR format) provided by the user. 
"""
function  MatCreateSeqBAIJWithArrays(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})
    Mat = PetscMat()
    error = MatCreateSeqBAIJWithArrays!(comm, bs, m, n, i, j, a, Mat)
    @assert iszero(error)
    return Mat
end

"""
    function  MatCreateSeqSBAIJWithArrays!(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)

Creates a sequential symmetric block AIJ matrix using 
matrix elements (in CSR format) provided by the user. 
"""
function  MatCreateSeqSBAIJWithArrays!(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)
    @check_if_loaded
    @check_if_initialized
    error = ccall( MatCreateSeqSBAIJWithArrays_ptr[],
            PetscErrorCode,
                (MPI.Comm,
                PetscInt,
                PetscInt,
                PetscInt,
                Ptr{PetscInt},
                Ptr{PetscInt},
                Ptr{PetscScalar},
                Ptr{Cvoid}),
            comm, bs, m, n, i, j, a, mat.mat)
    return error
end

"""
    function  MatCreateSeqSBAIJWithArrays(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})

Returns a sequential symmetric block AIJ matrix using 
matrix elements (in CSR format) provided by the user. 
"""
function  MatCreateSeqSBAIJWithArrays(
        comm::MPI.Comm,
        bs::Int,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})
    Mat = PetscMat()
    error = MatCreateSeqSBAIJWithArrays!(comm, bs, m, n, i, j, a, Mat)
    @assert iszero(error)
    return Mat
end

"""
    function MatDestroy!(mat::PetscMat)

Frees space taken by a matrix. 
"""
function MatDestroy!(mat::PetscMat)
    @check_if_loaded
    @check_if_initialized
    error = ccall( MatDestroy_ptr[],
            PetscErrorCode, 
                (Ptr{Cvoid},), 
            mat.mat)
    return error
end

"""
    function MatView(mat::PetscMat, viewer::PetscViewer=C_NULL)

Visualizes a matrix object. 
"""
function MatView(mat::PetscMat, viewer::PetscViewer=C_NULL)
    @check_if_loaded
    @check_if_initialized
    error = ccall( MatView_ptr[],
            PetscErrorCode, 
                (Ptr{Cvoid}, 
                Ptr{Cvoid}), 
            mat.mat[], viewer);
    return error
end



