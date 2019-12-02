
struct PETScSolver <: LinearSolver
    ksp :: PetscKSP
    function PETScSolver(Comm::MPI.Comm)
        ksp = KSPCreate(Comm)
        new(ksp)
    end
end

PETScSolver() = PETScSolver(MPI.COMM_SELF)


struct PETScSymbolicSetup <: SymbolicSetup 
    solver :: PETScSolver
end

struct PETScNumericalSetup <: NumericalSetup
    mat :: PetscMat
    solver :: PETScSolver
end

function symbolic_setup(
        ps::PETScSolver, 
        mat::AbstractSparseMatrix)
    return PETScSymbolicSetup(ps)
end

function numerical_setup!(
        pns::PETScNumericalSetup, 
        mat::PetscMat)
    if pns.mat.mat[] != mat.mat[] 
        error = MatDestroy!(pns.mat)
        @assert iszero(error)
        pns.mat.mat[] = mat.mat[]
        error = KSPSetOperators!(pns.solver.ksp, mat, mat)
        @assert iszero(error)
        error = KSPSetFromOptions!(pns.solver.ksp)
        @assert iszero(error)
        error = KSPSetUp!(pns.solver.ksp)
        @assert iszero(error)
    end
    return pns
end

function numerical_setup!(
        pns::PETScNumericalSetup, 
        mat::SparseMatrixCSR{0,PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqBAIJWithArrays(MPI.COMM_SELF, 1, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    return numerical_setup!(pns, Mat)
end

function numerical_setup!(
        pns::PETScNumericalSetup, 
        mat::SymSparseMatrixCSR{0,PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqSBAIJWithArrays(MPI.COMM_SELF, 1, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    return numerical_setup!(pns, Mat)
end

function numerical_setup(
        pss::PETScSymbolicSetup, 
        mat::SparseMatrixCSR{0,PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqBAIJWithArrays(MPI.COMM_SELF, 1, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    return numerical_setup!(PETScNumericalSetup(Mat, pss.solver), Mat)
end

function numerical_setup(
        pss::PETScSymbolicSetup, 
        mat::SymSparseMatrixCSR{0,PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqSBAIJWithArrays(MPI.COMM_SELF, 1, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    return numerical_setup!(PETScNumericalSetup(Mat, pss.solver), Mat)
end

function solve!(
        x::Vector{PetscScalar}, 
        ns::PETScNumericalSetup, 
        b::Vector{PetscScalar})
    B = VecCreateSeqWithArray(MPI.COMM_SELF, 1, length(b), b)
    X = VecCreateSeqWithArray(MPI.COMM_SELF, 1, length(x), x)
    error = KSPSolve!(ns.solver.ksp, B, X)
    @assert iszero(error)
    error = VecDestroy!(B)
    @assert iszero(error)
    error = VecDestroy!(X)
    @assert iszero(error)
end

