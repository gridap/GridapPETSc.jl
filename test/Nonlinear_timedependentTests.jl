using Gridap
using GridapPETSc
using LineSearches: BackTracking
using BenchmarkTools

options = "-snes_type newtonls -snes_linesearch_type basic -snes_monitor -ksp_converged_reason -snes_converged_reason -snes_linesearch_damping 1.0 -snes_rtol 1.0e-14  -pc_type ilu -ksp_type gmres "
𝒯 = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω,2)

refFE = ReferenceFE(lagrangian,Float64,1)

V = TestFESpace(𝒯,refFE,dirichlet_tags="boundary")

g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V,g)

κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,u,v) = ∫( ∂t(u)*v + κ(t)*(∇(u)⋅∇(v)) - f(t)*v )dΩ
jac(t,u,du,v) = ∫( κ(t)*(∇(du)⋅∇(v)) )dΩ
jac_t(t,u,duₜ,v) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)

Δt = 0.05
θ = 1.0

u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0

function main_petsc() 
    GridapPETSc.with(args=split(options)) do

nls = PETScNonlinearSolver()

ode_solver = ThetaMethod(nls,Δt,θ)

uₕₜ = solve(ode_solver,op,u₀,t₀,T)


createpvd("poisson_transient_solution_nlspetsc") do pvd
  for (uₕ,t) in uₕₜ
    pvd[t] = createvtk(Ω,"poisson_transient_solution_nlspetsc_$t"*".vtu",cellfields=["u"=>uₕ])
  end
end

end
end


function main_nls() 
nls = NLSolver( show_trace=true, method=:newton, linesearch=BackTracking())
ode_solver = ThetaMethod(nls,Δt,θ)
uₕₜ = solve(ode_solver,op,u₀,t₀,T)


createpvd("poisson_transient_solution_nls") do pvd
  for (uₕ,t) in uₕₜ
    pvd[t] = createvtk(Ω,"poisson_transient_solution_nls_$t"*".vtu",cellfields=["u"=>uₕ])
  end
end

end




main_petsc() 
main_nls()