using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using PartitionedArrays
using Test

function main(parts)
  options = "-info -pc_type jacobi -ksp_type cg -ksp_monitor -ksp_rtol 1.0e-12"
  GridapPETSc.with(args=split(options)) do
      domain = (0,4,0,4)
      cells = (4,4)
      model = CartesianDiscreteModel(parts,domain,cells)

      labels = get_face_labeling(model)
      add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
      add_tag_from_tags!(labels,"neumann",[4,6,8])

      Ω = Triangulation(model)
      Γn = Boundary(model,tags="neumann")
      n_Γn = get_normal_vector(Γn)

      u(x) = x[1] + x[2]
      f(x) = -Δ(u,x)

      reffe = ReferenceFE(lagrangian,Float64,1)
      V = TestFESpace(model,reffe,dirichlet_tags="boundary")
      U = TrialFESpace(u,V)

      dΩ = Measure(Ω,2)
      dΓn = Measure(Γn,2)

      function a(u,v)
        ∫(∇(v)⋅∇(u))dΩ
      end
      function l(v)
        ∫(v*f)dΩ
      end
      op = AffineFEOperator(a,l,U,V)

      ls = PETScLinearSolver()
      fels = LinearFESolver(ls)
      uh = solve(fels,op)
      eh = u - uh
      @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
  end
 end
