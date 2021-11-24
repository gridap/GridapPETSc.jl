using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using PartitionedArrays
using Test


# GridapDistributedPETScWrappers.C.KSPCreate(comm(A),ksp)
# GridapDistributedPETScWrappers.C.KSPSetOperators(ksp[],A.p,A.p)
# GridapDistributedPETScWrappers.C.KSPSetType(ksp[],GridapDistributedPETScWrappers.C.KSPPREONLY)
# GridapDistributedPETScWrappers.C.KSPGetPC(ksp[],pc)

# # If system is SPD use the following two calls
# GridapDistributedPETScWrappers.C.PCSetType(pc[],GridapDistributedPETScWrappers.C.PCCHOLESKY)
# GridapDistributedPETScWrappers.C.MatSetOption(A.p,
#                                               GridapDistributedPETScWrappers.C.MAT_SPD,GridapDistributedPETScWrappers.C.PETSC_TRUE);
# # Else ... use only the following one
# # GridapDistributedPETScWrappers.C.PCSetType(pc,GridapDistributedPETScWrappers.C.PCLU)

# PCFactorSetMatSolverType(pc[],GridapDistributedPETScWrappers.C.MATSOLVERMUMPS)
# PCFactorSetUpMatSolverType(pc[])
# GridapDistributedPETScWrappers.C.PCFactorGetMatrix(pc[],mumpsmat)
# MatMumpsSetIcntl(mumpsmat[],4 ,2)     # level of printing (0 to 4)
# MatMumpsSetIcntl(mumpsmat[],28,2)     # use 1 for sequential analysis and ictnl(7) ordering,
#                                     # or 2 for parallel analysis and ictnl(29) ordering
# MatMumpsSetIcntl(mumpsmat[],29,2)     # parallel ordering 1 = ptscotch, 2 = parmetis
# MatMumpsSetCntl(mumpsmat[] ,3,1.0e-6)  # threshhold for row pivot detection


# Setup solver via low level PETSC API calls
function mykspsetup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

function main(parts)
  options = "-info -ksp_type preonly -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type mumps"
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

      k = 2
      u((x,y)) = (x+y)^k
      f(x) = -Δ(u,x)
      g = n_Γn⋅∇(u)

      reffe = ReferenceFE(lagrangian,Float64,k)
      V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
      U = TrialFESpace(u,V)

      dΩ = Measure(Ω,2*k)
      dΓn = Measure(Γn,2*k)

      a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
      l(v) = ∫( v*f )dΩ + ∫( v*g )dΓn
      op = AffineFEOperator(a,l,U,V)

      ls = PETScLinearSolver(mykspsetup)
      fels = LinearFESolver(ls)
      uh = solve(fels,op)
      eh = u - uh
      @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
  end
 end
