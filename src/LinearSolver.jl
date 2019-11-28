
struct PETScSolver <: LinearSolver
    ksp :: PetscKSP
    function PETScSolver(Comm::MPI.Comm)
        ksp = KSPCreate(Comm)
        new(ksp)
    end
end

PETScSolver() = PETScSolver(MPI.COMM_SELF)


struct PETScSymbolicSetup <: SymbolicSetup 
    mat :: PetscMat
    solver :: PETScSolver
end


function PETScSymbolicSetup(mat::AbstractSparseMatrix{PetscScalar,PetscInt}, solver::PETScSolver)
    m, n = size(mat)
    Mat = MatCreateSeqAIJWithArrays(MPI.COMM_SELF, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    PetscSymbolicSetyp(Mat, solver)
end

struct PETScNumericalSetup <: NumericalSetup
    mat :: PetscMat
    solver :: PETScSolver
end

function symbolic_setup(
        ps::PETScSolver, 
        mat::AbstractSparseMatrix{PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqAIJWithArrays(MPI.COMM_SELF, m, n, getptr(mat), getindices(mat), nonzeros(mat))
    error = KSPSetOperators!(ps.ksp, Mat, Mat)
    @assert iszero(error)
    return PETScSymbolicSetup(Mat, ps)
end

function numerical_setup!(
        pns::PETScNumericalSetup, 
        mat::PetscMat)
    return pns
end

function numerical_setup(
        pss::PETScSymbolicSetup, 
        mat::AbstractSparseMatrix{PetscScalar,PetscInt})
    m, n = size(mat)
    Mat = MatCreateSeqAIJWithArrays(MPI.COMM_SELF, m, n, getptr(mat), getindices(mat), nonzeros(mat))
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

