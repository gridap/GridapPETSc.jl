module PETScNonlinearSolversTests

using Gridap
using GridapPETSc
using Test

options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 1.0 -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -pc_type cholesky -ksp_type preonly -snes_converged_reason"
GridapPETSc.Init(args=split(options))

op  = Gridap.Algebra.NonlinearOperatorMock()
nls = PETScNonlinearSolver()

x0 = zero_initial_guess(op)
x = [1.0, 3.0]
Gridap.Algebra.test_nonlinear_solver(nls,op,x0,x)

x0 = [2.1,2.9]
x = [2.0, 3.0]
Gridap.Algebra.test_nonlinear_solver(nls,op,x0,x)

x0 = zero_initial_guess(op)
cache = solve!(x0,nls,op)

GridapPETSc.destroy(cache)
GridapPETSc.Finalize()

end # module
