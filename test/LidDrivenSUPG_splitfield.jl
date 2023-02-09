using Revise
using Gridap
using Gridap.CellData


using Test
using GridapDistributed
using GridapDistributed.CellData


using PartitionedArrays
using MPI
using SparseArrays
using SparseMatricesCSR
using GridapPETSc
using BenchmarkTools

# options = "-snes_type newtonls -snes_linesearch_type basic -snes_linesearch_damping 1.0 -snes_rtol 1.0e-10 -snes_atol 0.0 -snes_monitor -log_view \
# -pc_use_amat -ksp_type fgmres  -ksp_converged_reason -ksp_max_it 50 -ksp_rtol 1e-3 -ksp_atol 1e-8 \
# -fieldsplit_vel_pc_type ilu -fieldsplit_vel_pc_factor_levels 2 -fieldsplit_velu_ksp_type gmres -fieldsplit_vel_ksp_monitor -fieldsplit_vel_ksp_converged_reason \
# -fieldsplit_pres_pc_type gamg -fieldsplit_pres_ksp_type gmres -fieldsplit_pres_ksp_monitor fieldsplit_pres_ksp_converged_reason" #Very good pressure convergence

# options = "-snes_type newtonls -snes_linesearch_type basic -snes_linesearch_damping 1.0 -snes_rtol 1.0e-10 -snes_atol 0.0 -snes_monitor -log_view \
# -ksp_type fgmres  -ksp_converged_reason -ksp_max_it 50 -ksp_rtol 1e-3 -ksp_atol 1e-8 -ksp_monitor_short \
# -fieldsplit_vel_pc_type asm -fieldsplit_vel_ksp_type gmres -fieldsplit_vel_ksp_converged_reason \
# -fieldsplit_pres_pc_type gamg -fieldsplit_pres_ksp_type gmres -fieldsplit_pres_ksp_converged_reason" #Works


options = "-snes_type newtonls -snes_linesearch_type basic -snes_linesearch_damping 1.0 -snes_rtol 1.0e-6 -snes_atol 0.0 -snes_monitor -log_view \
-ksp_type fgmres  -ksp_converged_reason -ksp_max_it 50 -ksp_rtol 1e-3 -ksp_atol 1e-10 -ksp_monitor_short \
-fieldsplit_vel_pc_type gamg -fieldsplit_vel_ksp_type gmres -fieldsplit_vel_ksp_converged_reason \
-fieldsplit_pres_pc_type gamg -fieldsplit_pres_ksp_type gmres -fieldsplit_pres_ksp_converged_reason" #Works


#-fieldsplit_vel_ksp_converged_reason -ksp_monitor_short
#-fieldsplit_pres_ksp_converged_reason
#-log_view -mat_view :lid.m:ascii_matlab
function stretching_y_function(x)
    gamma1 = 2.5
    S = 0.5815356159649889 #for rescaling the function over the domain -0.5 -> 0.5
    -tanh.(gamma1 .* (x)) ./ tanh.(gamma1) .* S
end


function stretching(x::Point)
    m = zeros(length(x))
    m[1] = stretching_y_function(x[1])


    m[2] = stretching_y_function(x[2])
    Point(m)
end

function main_ld(parts)
t0 = 0
dt = 0.01
tF = 0.05
θ = 1
θvp = 1
ν = 0.0001

n = 256
L = 0.5
domain = (-L, L, -L, L)
partition = (n,n)
model = CartesianDiscreteModel(parts, domain, partition,  map=stretching)

labels = get_face_labeling(model)
add_tag_from_tags!(labels, "diri1", [5,])
add_tag_from_tags!(labels, "diri0", [1, 2, 4, 3, 6, 7, 8])
add_tag_from_tags!(labels, "p", [4,])

u_wall= VectorValue(0.0, 0.0) 
u_top = VectorValue(1.0, 0.0)

u_diri_values = [u_wall, u_top]


order = 1
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order)

V = TestFESpace(model,reffeᵤ,dirichlet_tags=["diri0","diri1"],conformity=:H1)
Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags="p")
Y = MultiFieldFESpace([V,Q])
U = TrialFESpace(V, u_diri_values)
P = TrialFESpace(Q,0.0)
X = MultiFieldFESpace([U,P])

degree = order*4
Ω= Triangulation(model)
dΩ = Measure(Ω,degree)

h = h_param(Ω,2)
function τ(u, h)
    r = 1
    τ₂ = h^2 / (4 * ν)
    val(x) = x
    val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
    u = val(norm(u))

    if iszero(u)
        return τ₂
    end

    τ₃ = dt / 2
    τ₁ = h / (2 * u)
    return 1 / (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r)


end


τb(u, h) = (u ⋅ u) * τ(u, h)


Rm(t, (u, p)) = ∂t(u) + u ⋅ ∇(u) + ∇(p) #- hf(t)  #- ν*Δ(u)
Rc(u) = ∇ ⋅ u
dRm((u, p), (du, dp), (v, q)) = du ⋅ ∇(u) + u ⋅ ∇(du) + ∇(dp) #- ν*Δ(du)
dRc(du) = ∇ ⋅ du

Bᴳ(t, (u, p), (v, q)) = ∫(∂t(u) ⋅ v)dΩ + ∫((u ⋅ ∇(u)) ⋅ v)dΩ - ∫((∇ ⋅ v) * p)dΩ + ∫((q * (∇ ⋅ u)))dΩ + ν * ∫(∇(v) ⊙ ∇(u))dΩ #- ∫(hf(t) ⋅ v)dΩ
B_stab(t, (u, p), (v, q)) = ∫((τ ∘ (u.cellfield, h) * (u ⋅ ∇(v) + ∇(q))) ⊙ Rm(t, (u, p)) # First term: SUPG, second term: PSPG u⋅∇(v) + ∇(q)
+
τb ∘ (u.cellfield, h) * (∇ ⋅ v) ⊙ Rc(u) # Bulk viscosity. Try commenting out both stabilization terms to see what happens in periodic and non-periodic cases
)dΩ
res(t, (u, p), (v, q)) = Bᴳ(t, (u, p), (v, q)) + B_stab(t, (u, p), (v, q))

dBᴳ(t, (u, p), (du, dp), (v, q)) = ∫(((du ⋅ ∇(u)) ⋅ v) + ((u ⋅ ∇(du)) ⋅ v) + (∇(dp) ⋅ v) +  (q * (∇ ⋅ du)))dΩ + ν * ∫(∇(v) ⊙ ∇(du))dΩ
dB_stab(t, (u, p), (du, dp), (v, q)) = ∫(((τ ∘ (u.cellfield, h) * (u ⋅ ∇(v)' +  ∇(q))) ⊙ dRm((u, p), (du, dp), (v, q))) + ((τ ∘ (u.cellfield, h) * (du ⋅ ∇(v)')) ⊙ Rm(t, (u, p))) + (τb ∘ (u.cellfield, h) * (∇ ⋅ v) ⊙ dRc(du)))dΩ


jac(t, (u, p), (du, dp), (v, q)) = dBᴳ(t, (u, p), (du, dp), (v, q)) + dB_stab(t, (u, p), (du, dp), (v, q))

jac_t(t, (u, p), (dut, dpt), (v, q)) = ∫(dut ⋅ v)dΩ + ∫(τ ∘ (u.cellfield, h) * (u ⋅ ∇(v) + (θvp)*∇(q)) ⊙ dut)dΩ

op = TransientFEOperator(res, jac, jac_t, X,Y)


xh0 = zero(X)


#stokes_pc_fieldsplit_off_diag_use_amat -stokes_ksp_type pipegcr
#GridapPETSc.Init(args=split(options))
GridapPETSc.with(args=split(options)) do

#Definitions of IS
U_Parray, P_Parray = field_dof_split(Y,U,P)

ISVel = PETScIS(U_Parray);
ISP = PETScIS(P_Parray);

@check_error_code GridapPETSc.PETSC.ISView(ISVel.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD)
@check_error_code GridapPETSc.PETSC.ISView(ISP.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD)

#@check_error_code GridapPETSc.PETSC.ISView(PETScIS(collect(1:5)).is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD)

function mysnessetup(snes)
ksp      = Ref{GridapPETSc.PETSC.KSP}()
pc       = Ref{GridapPETSc.PETSC.PC}()
@check_error_code GridapPETSc.PETSC.SNESSetFromOptions(snes[])
@check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
#@check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
#@check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPFGMRES)
@check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
@check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCFIELDSPLIT)
@check_error_code GridapPETSc.PETSC.PCFieldSplitSetType(pc[],GridapPETSc.PETSC.PC_COMPOSITE_ADDITIVE)
@check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
@check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
#After KSP setFromOption
@check_error_code GridapPETSc.PETSC.PCFieldSplitSetIS(pc[],"vel",ISVel.is[])
@check_error_code GridapPETSc.PETSC.PCFieldSplitSetIS(pc[],"pres",ISP.is[])

end

function solve_sim(solver_type)
    if solver_type == :petsc
        solver = PETScNonlinearSolver(mysnessetup)
    elseif solver_type == :julia
        solver = NLSolver(show_trace=true, method=:newton)
    end

    ode_solver = ThetaMethod(solver, dt, θ)

    solt = solve(ode_solver, op, xh0, t0, tF)
    iter = 0
    @time createpvd(parts,"LidDriven") do pvd
    for (xh_tn, tn) in solt
                println("iteration = $iter")
                iter = iter +1
                uh_tn = xh_tn[1]
                ph_tn = xh_tn[2]
                ωh_tn = ∇ × uh_tn
            pvd[tn] = createvtk(Ω, "LidDriven_$tn"*".vtu", cellfields=["uh" => uh_tn, "ph" => ph_tn, "wh" => ωh_tn])
    end
    end
end

solve_sim(:petsc)
solve_sim(:julia)




#GridapPETSc.Finalize()
end
end


function field_dof_split(Y,U,P)
    xrows = Y.gids.partition
    urows = U.gids.partition
    prows = P.gids.partition
    
    ulrows,plrows = map_parts(urows,prows,xrows) do urows,prows,xrows
            uloc_idx = collect(1:1:length(urows.oid_to_lid))
            ploc_idx = collect(length(uloc_idx)+1 :1: length(xrows.oid_to_lid))
            ur = xrows.lid_to_gid[xrows.oid_to_lid][uloc_idx]
            pr = xrows.lid_to_gid[xrows.oid_to_lid][ploc_idx]
    
            return ur, pr
    end
    ur = ulrows.part .-1
    pr = plrows.part .-1

    return ur,pr
end

function h_param(Ω::GridapDistributed.DistributedTriangulation, D::Int64)
    h = map_parts(Ω.trians) do trian
        h_param(trian, D)
    end
    h = CellData.CellField(h, Ω)
    h
end


function h_param(Ω::Triangulation, D::Int64)
    h = lazy_map(h -> h^(1 / D), get_cell_measure(Ω))

    h
end

partition =(2,2)
with_backend(main_ld, MPIBackend(), partition)

#mpiexecjl --project=. -n 4 julia LidDrivenSUPG_IS_distributed.jl
