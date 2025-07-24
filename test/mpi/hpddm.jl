using Gridap
using GridapDistributed
using PartitionedArrays
using SparseMatricesCSR, SparseArrays
using GridapPETSc

u(x) = x[1] + x[2]

np = (2,2)
ranks = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(8,8))

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V,u)

qdegree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

f(x) = -Δ(u)(x)
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
l(v) = ∫(f⋅v)dΩ

Ωg = Triangulation(with_ghost,model)
dΩg = Measure(Ωg,qdegree)
a_g(u,v) = ∫(∇(u)⋅∇(v))dΩg

assem = SparseMatrixAssembler(
  SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},U,V
)
op = AffineFEOperator(a,l,U,V,assem)

options = "-ksp_error_if_not_converged true -ksp_converged_reason -ksp_monitor -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_define_subdomains"
GridapPETSc.with(args=split(options)) do
  solver = HPDDMLinearSolver(V,a_g)
  uh = solve(solver,op)

  eh = u - uh
  err_l2 = sqrt(sum(∫(eh⋅eh)dΩ))
  if i_am_main(ranks)
    @info "L2 error: $err_l2"
  end
end
