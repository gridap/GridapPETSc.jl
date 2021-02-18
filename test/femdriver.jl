module FEMDriver

using Test
using Gridap
using GridapPETSc

tol = 1e-10
maxits = 1000
GridapPETSc.Init(["-ksp_type", "cg",
                  "-ksp_monitor",
                  "-ksp_rtol", "$tol",
                  "-ksp_converged_reason",
                  "-ksp_max_it", "$maxits",
                  "-ksp_norm_type", "unpreconditioned",
                  "-ksp_view",
                  "-pc_type","gamg",
                  "-pc_gamg_type","agg",
                  "-pc_gamg_est_ksp_type","cg",
                  "-mg_levels_esteig_ksp_type","cg",
                  "-mg_coarse_sub_pc_type","cholesky",
                  "-mg_coarse_sub_pc_factor_mat_ordering_type","nd",
                  "-pc_gamg_process_eq_limit","50",
                  "-pc_gamg_square_graph","0",
                  "-pc_gamg_agg_nsmooths","1",
                  "-build_twosided","redscatter"])

domain = (0,1,0,1,0,1)
cells  = (10,10,10)
model  = CartesianDiscreteModel(domain,cells)

order = 1
V = TestFESpace( model,
      ReferenceFE(lagrangian,Float64,order),
      conformity=:H1, dirichlet_tags="boundary" )
U = TrialFESpace(V)

Ω = Triangulation(model)

degree = 2*order
dΩ = Measure(Ω,degree)

f(x) = x[1]*x[2]

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
l(v) = ∫( v*f )*dΩ

ass = SparseMatrixAssembler(SparseMatrixCSR{0,PetscReal,PetscInt},U,V)
op = AffineFEOperator(a,l,ass)

ls = PETScSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

GridapPETSc.Finalize()

end #module
