
struct PetscMat
    mat::Ref{Ptr{Cvoid}}
    function PetscMat()
        mat = Ref{Ptr{Cvoid}}()
        new(mat)
    end
end

"""
    function  MatCreateSeqAIJWithArrays!(
        comm::MPI.Comm,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)

Creates a sequential AIJ matrix using matrix elements
(in CSR format) provided by the user.
"""
function  MatCreateSeqAIJWithArrays!(
        comm::MPI.Comm,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar},
        mat::PetscMat)
    @check_if_loaded
    @check_if_initialized
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

"""
    function  MatCreateSeqAIJWithArrays(
        comm::MPI.Comm,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})

Returns a sequential AIJ matrix using matrix elements
(in CSR format) provided by the user.
"""
function  MatCreateSeqAIJWithArrays(
        comm::MPI.Comm,
        m::Int,
        n::Int,
        i::Vector{PetscInt},
        j::Vector{PetscInt},
        a::Vector{PetscScalar})
    Mat = PetscMat()
    error = MatCreateSeqAIJWithArrays!(comm, m, n, i, j, a, Mat)
    @assert iszero(error)
    return Mat
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
    function MatGetSize(A::PetscMat)

Returns the numbers of rows and columns in a matrix.
"""
function MatGetSize(A::PetscMat)
    @check_if_loaded
    @check_if_initialized
    m = Vector{PetscInt}(undef,1)
    n = Vector{PetscInt}(undef,1)
    error = ccall( MatGetSize_ptr[],
            PetscErrorCode,
                (Ptr{Cvoid},
                Ptr{PetscInt},
                Ptr{PetscInt}),
            A.mat[], m, n)
    @assert iszero(error)
    return (m[1], n[1])
end

"""
    function MatEqual!(A::PetscMat, B::PetscMat)

Compare two matrices.
"""
function MatEqual(A::PetscMat, B::PetscMat)
    @check_if_loaded
    @check_if_initialized
    is_equal = Vector{PetscBool}(undef,1)
    error = ccall( MatEqual_ptr[],
            PetscErrorCode,
                (Ptr{Cvoid},
                Ptr{Cvoid},
                Ptr{PetscBool}),
            A.mat[], B.mat[], is_equal)
    @assert iszero(error)
    return is_equal[1] == PETSC_TRUE
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
