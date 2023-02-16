module StokesTest

using Gridap
using Test
using GridapDistributed

using PartitionedArrays
using SparseArrays
using GridapPETSc

#You can provide the following options string (which will be automatically used when calling PetscLinearSolver())
# or the PescLinearSolver(mykspsetup)
#They are equivalent (with the exeption of -ksp_monitor)
#You can use PescLinearSolver(mykspsetup) and at the same time "-ksp_monitor", they are merged

function mykspsetup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()
    umfpack = Ref{GridapPETSc.PETSC.Mat}()
    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPPREONLY)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCLU)
    @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[], GridapPETSc.PETSC.MATSOLVERUMFPACK)
    @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
    @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[], umfpack)
    @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
end


function main(solver_options)
    if solver_options == :petsc_linecommand
        options = "-pc_type lu -ksp_type preonly -ksp_max_it 10 -ksp_monitor -pc_factor_mat_solver_type umfpack"
    elseif solver_options == :petsc_mykspsetup
        options = "-ksp_monitor"
    elseif solver_options == :julia
        options = " "
    else
        error("Solver $(solver_options) not valid. Use instead:\n\
         :petsc_linecommand\n\
         :petsc_mykspsetup\n\
         :julia\n")
    end
    tt = 0.0
    GridapPETSc.with(args=split(options)) do
        n = 50
        domain = (0, 1, 0, 1)
        partition = (n, n)
        model = CartesianDiscreteModel(domain, partition)

        labels = get_face_labeling(model)
        add_tag_from_tags!(labels, "diri1", [6,])
        add_tag_from_tags!(labels, "diri0", [1, 2, 3, 4, 5, 7, 8])
        add_tag_from_tags!(labels, "dirip", [1])

        order = 2
        reffeᵤ = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
        reffeₚ = ReferenceFE(lagrangian, Float64, order - 1; space=:P)

        V = TestFESpace(model, reffeᵤ, labels=labels, dirichlet_tags=["diri0", "diri1"], conformity=:H1)
        Q = TestFESpace(model, reffeₚ, conformity=:L2, dirichlet_tags="dirip")
        Y = MultiFieldFESpace([V, Q])

        u0 = VectorValue(0, 0)
        u1 = VectorValue(1, 0)
        U = TrialFESpace(V, [u0, u1])
        P = TrialFESpace(Q, 0.0)
        X = MultiFieldFESpace([U, P])

        degree = order
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)

        f = VectorValue(0.0, 0.0)
        h = 1 / n
        τ = (h .^ 2) ./ 4

        a((u, p), (v, q)) = ∫(∇(v) ⊙ ∇(u) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ + ∫((τ ⋅ ∇(q))' ⋅ (∇(p)))dΩ
        #The last term is added in order to have elements on the main diagonal, if not PETSc does not work properly

        l((v, q)) = ∫(v ⋅ f)dΩ
        res((u, p), (v, q)) = a((u, p), (v, q)) - l((v, q))
        op = AffineFEOperator(a, l, X, Y)


        if solver_options == :petsc_linecommand
            #Solve using Petsc solver, with the options given in line command style
            solver = PETScLinearSolver()
            uh, ph = solve(solver, op)
            tt = @elapsed uh, ph = solve(solver, op)
        elseif solver_options == :petsc_mykspsetup
            #Solve using Petsc solver, with the options given in  mykspsetup
            solver = PETScLinearSolver(mykspsetup)
            uh, ph = solve(solver, op)
            tt = @elapsed uh, ph = solve(solver, op)
        elseif solver_options == :julia
            #Solve using default julia solver, which is a LU decomposition
            uh, ph = solve(op)
            tt = @elapsed uh, ph = solve(op)
        end

    end
    return tt

end
#The PETSc and Julia in this case both use a LU decomposition, the time should be really close

j = main(:julia)
pl = main(:petsc_linecommand)
pm = main(:petsc_mykspsetup)

@test isapprox(j,pl; rtol = 0.1)
@test isapprox(j,pm; rtol = 0.1)
@test isapprox(pm,pm; rtol = 0.1)

end #end module
