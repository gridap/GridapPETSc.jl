"""
Low level interface with PETSC, which serve as the back-end in GridapPETSc.

The types and functions defined here are almost 1-to-1 to the corresponding C counterparts. In particular, the types defined
can be directly used to call C PETSc routines via `ccall`.
When a C function expects a pointer, use a `Ref` to the corresponding Julia alias.
E.g., if an argument is `PetscBool *` in the C code, pass an object with type `Ref{PetscBool}` from the Julia code.
Using this rule, [`PETSC.PetscInitialized`](@ref) can be called as

    flag = Ref{PetscBool}()
    @check_error_code PetscInitialized(flag)
    if flag[] == PETSC_TRUE
      println("Petsc is initialized!")
    end
"""
module PETSC

using Libdl
using GridapPETSc: libpetsc_handle
using MPI

include("Config.jl")

let types_jl = joinpath(@__DIR__,"..","deps","PetscDataTypes.jl")
  if !isfile(types_jl)
    msg = """
    GridapPETSc needs to be configured before use. Type
  
    pkg> build
  
    and try again.
    """
    error(msg)
  end
  
  include(types_jl)
end

macro wrapper(fn,rt,argts,args,url)
  sargs = "$(args)"
  if length(args.args) == 1
    sargs = sargs[1:end-2]*")"
  end
  if isempty(rstrip(url))
    str = """
        $(fn.value)$(sargs)
    """
  else
    str = """
        $(fn.value)$(sargs)

    See [PETSc manual]($url).
    """
  end
  expr = quote
    @doc $str
    function $(fn.value)($(args.args...))
      ccall(
        Libdl.dlsym(libpetsc_handle[],$fn),
        $rt,$argts,$(args.args...))
    end
  end
  esc(expr)
end

#Petsc init related functions

@wrapper(:PetscInitializeNoArguments,PetscErrorCode,(),(),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatEqual.html")
@wrapper(:PetscInitializeNoPointers,PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),(argc,args,filename,help),"")
@wrapper(:PetscFinalize,PetscErrorCode,(),(),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalize.html")
@wrapper(:PetscFinalized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscFinalized.html")
@wrapper(:PetscInitialized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitialized.html")

# viewer related functions

"""
Julia alias for `PetscViewer` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PetscViewer.html).
"""
struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)

@wrapper(:PETSC_VIEWER_STDOUT_,PetscViewer,(MPI.Comm,),(comm,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_.html")
@wrapper(:PETSC_VIEWER_DRAW_,PetscViewer,(MPI.Comm,),(comm,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_.html")

"""
    @PETSC_VIEWER_STDOUT_SELF

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_SELF.html).
"""
macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end 

"""
    @PETSC_VIEWER_STDOUT_WORLD

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_STDOUT_WORLD.html).
"""
macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end 

"""
    @PETSC_VIEWER_DRAW_SELF

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_SELF.html).
"""
macro PETSC_VIEWER_DRAW_SELF()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_SELF)
  end
end 

"""
    @PETSC_VIEWER_DRAW_WORLD

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/PETSC_VIEWER_DRAW_WORLD.html).
"""
macro PETSC_VIEWER_DRAW_WORLD()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_WORLD)
  end
end 

# Vector related functions

"""
Julia alias for the `InsertMode` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/InsertMode.html).
"""
@enum InsertMode begin
  NOT_SET_VALUES
  INSERT_VALUES
  ADD_VALUES
  MAX_VALUES
  MIN_VALUES
  INSERT_ALL_VALUES
  ADD_ALL_VALUES
  INSERT_BC_VALUES
  ADD_BC_VALUES
end

"""
Julia alias for the `Vec` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/Vec.html).
"""
struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)

@wrapper(:VecCreateSeq,PetscErrorCode,(MPI.Comm,PetscInt,Ptr{Vec}),(comm,n,vec),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeq.html")
@wrapper(:VecCreateSeqWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,array,vec),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeqWithArray.html")
@wrapper(:VecDestroy,PetscErrorCode,(Ptr{Vec},),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDestroy.html")
@wrapper(:VecView,PetscErrorCode,(Vec,PetscViewer),(vec,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecView.html")
@wrapper(:VecSetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(x,ni,ix,y,iora),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetValues.html")
@wrapper(:VecGetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(x,ni,ix,y),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetValues.html")
@wrapper(:VecAssemblyBegin,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyBegin.html")
@wrapper(:VecAssemblyEnd,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyEnd.html")
@wrapper(:VecPlaceArray,PetscErrorCode,(Vec,Ptr{PetscScalar}),(vec,array),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecPlaceArray.html")

# Matrix related functions


"""
Julia alias for the `MatAssemblyType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyType.html#MatAssemblyType).
"""
@enum MatAssemblyType begin
  MAT_FINAL_ASSEMBLY=0
  MAT_FLUSH_ASSEMBLY=1
end

"""
Julia constant storing the `PETSC_DEFAULT` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DEFAULT.html).
"""
const PETSC_DEFAULT = Cint(-2)

"""
Julia constant storing the `PETSC_DECIDE` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DECIDE.html).
"""
const PETSC_DECIDE = Cint(-1)

"""
Julia constant storing the `PETSC_DETERMINE` value.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_DETERMINE.html).
"""
const PETSC_DETERMINE = PETSC_DECIDE

"""
Julia alias for the `Mat` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/Mat.html).
"""
struct Mat
  ptr::Ptr{Cvoid}
end
Mat() = Mat(Ptr{Cvoid}())
Base.convert(::Type{Mat},p::Ptr{Cvoid}) = Mat(p)

@wrapper(:MatCreateSeqAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,nz,nnz,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJ.html")
@wrapper(:MatCreateSeqAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,i,j,a,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJWithArrays.html")
@wrapper(:MatDestroy,PetscErrorCode,(Ptr{Mat},),(A,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDestroy.html")
@wrapper(:MatView,PetscErrorCode,(Mat,PetscViewer),(mat,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatView.html")
@wrapper(:MatSetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(mat,m,idxm,n,idxn,v,addv),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValues.html")
@wrapper(:MatGetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(mat,m,idxm,n,idxn,v),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetValues.html")
@wrapper(:MatAssemblyBegin,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyBegin.html")
@wrapper(:MatAssemblyEnd,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyEnd.html")
@wrapper(:MatGetSize,PetscErrorCode,(Mat,Ptr{PetscInt},Ptr{PetscInt}),(mat,m,n),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetSize.html")
@wrapper(:MatEqual,PetscErrorCode,(Mat,Mat,Ptr{PetscBool}),(A,B,flg),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatEqual.html")

# KSP and PC related things

"""
Julia alias for `KSPType` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPType.html).
"""
const KSPType = Cstring
const KSPRICHARDSON = "richardson"
const KSPCHEBYSHEV  = "chebyshev"
const KSPCG         = "cg"
const KSPGROPPCG    = "groppcg"
const KSPPIPECG     = "pipecg"
const KSPPIPECGRR   = "pipecgrr"
const KSPPIPELCG    = "pipelcg"
const KSPPIPEPRCG   = "pipeprcg"
const KSPPIPECG2    = "pipecg2"
const KSPCGNE       = "cgne"
const KSPNASH       = "nash"
const KSPSTCG       = "stcg"
const KSPGLTR       = "gltr"
const KSPFCG        = "fcg"
const KSPPIPEFCG    = "pipefcg"
const KSPGMRES      = "gmres"
const KSPPIPEFGMRES = "pipefgmres"
const KSPFGMRES     = "fgmres"
const KSPLGMRES     = "lgmres"
const KSPDGMRES     = "dgmres"
const KSPPGMRES     = "pgmres"
const KSPTCQMR      = "tcqmr"
const KSPBCGS       = "bcgs"
const KSPIBCGS      = "ibcgs"
const KSPFBCGS      = "fbcgs"
const KSPFBCGSR     = "fbcgsr"
const KSPBCGSL      = "bcgsl"
const KSPPIPEBCGS   = "pipebcgs"
const KSPCGS        = "cgs"
const KSPTFQMR      = "tfqmr"
const KSPCR         = "cr"
const KSPPIPECR     = "pipecr"
const KSPLSQR       = "lsqr"
const KSPPREONLY    = "preonly"
const KSPQCG        = "qcg"
const KSPBICG       = "bicg"
const KSPMINRES     = "minres"
const KSPSYMMLQ     = "symmlq"
const KSPLCD        = "lcd"
const KSPPYTHON     = "python"
const KSPGCR        = "gcr"
const KSPPIPEGCR    = "pipegcr"
const KSPTSIRM      = "tsirm"
const KSPCGLS       = "cgls"
const KSPFETIDP     = "fetidp"
const KSPHPDDM      = "hpddm"

"""
Julia alias for `PCType` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCType.html).
"""
const PCType = Cstring
const PCNONE            = "none"
const PCJACOBI          = "jacobi"
const PCSOR             = "sor"
const PCLU              = "lu"
const PCSHELL           = "shell"
const PCBJACOBI         = "bjacobi"
const PCMG              = "mg"
const PCEISENSTAT       = "eisenstat"
const PCILU             = "ilu"
const PCICC             = "icc"
const PCASM             = "asm"
const PCGASM            = "gasm"
const PCKSP             = "ksp"
const PCCOMPOSITE       = "composite"
const PCREDUNDANT       = "redundant"
const PCSPAI            = "spai"
const PCNN              = "nn"
const PCCHOLESKY        = "cholesky"
const PCPBJACOBI        = "pbjacobi"
const PCVPBJACOBI       = "vpbjacobi"
const PCMAT             = "mat"
const PCHYPRE           = "hypre"
const PCPARMS           = "parms"
const PCFIELDSPLIT      = "fieldsplit"
const PCTFS             = "tfs"
const PCML              = "ml"
const PCGALERKIN        = "galerkin"
const PCEXOTIC          = "exotic"
const PCCP              = "cp"
const PCBFBT            = "bfbt"
const PCLSC             = "lsc"
const PCPYTHON          = "python"
const PCPFMG            = "pfmg"
const PCSYSPFMG         = "syspfmg"
const PCREDISTRIBUTE    = "redistribute"
const PCSVD             = "svd"
const PCGAMG            = "gamg"
const PCCHOWILUVIENNACL = "chowiluviennacl"
const PCROWSCALINGVIENNACL = "rowscalingviennacl"
const PCSAVIENNACL      = "saviennacl"
const PCBDDC            = "bddc"
const PCKACZMARZ        = "kaczmarz"
const PCTELESCOPE       = "telescope"
const PCPATCH           = "patch"
const PCLMVM            = "lmvm"
const PCHMG             = "hmg"
const PCDEFLATION       = "deflation"
const PCHPDDM           = "hpddm"
const PCHARA            = "hara"

"""
Julia alias for the `KSP` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSP.html).
"""
struct KSP
  ptr::Ptr{Cvoid}
end
KSP() = KSP(Ptr{Cvoid}())
Base.convert(::Type{KSP},p::Ptr{Cvoid}) = KSP(p)

"""
Julia alias for the `PC` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PC.html).
"""
struct PC
  ptr::Ptr{Cvoid}
end
PC() = PC(Ptr{Cvoid}())
Base.convert(::Type{PC},p::Ptr{Cvoid}) = PC(p)

@wrapper(:KSPCreate,PetscErrorCode,(MPI.Comm,Ptr{KSP}),(comm,inksp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPCreate.html")
@wrapper(:KSPDestroy,PetscErrorCode,(Ptr{KSP},),(ksp,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPDestroy.html")
@wrapper(:KSPSetFromOptions,PetscErrorCode,(KSP,),(ksp,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetFromOptions.html")
@wrapper(:KSPSetUp,PetscErrorCode,(KSP,),(ksp,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetUp.html")
@wrapper(:KSPSetOperators,PetscErrorCode,(KSP,Mat,Mat),(ksp,Amat,Pmat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetOperators.html")
@wrapper(:KSPSetTolerances,PetscErrorCode,(KSP,PetscReal,PetscReal,PetscReal,PetscInt),(ksp,rtol,abstol,dtol,maxits),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetTolerances.html")
@wrapper(:KSPSolve,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSolve.html")
@wrapper(:KSPSolveTranspose,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSolveTranspose.html")
@wrapper(:KSPGetIterationNumber,PetscErrorCode,(KSP,Ptr{PetscInt}),(ksp,its),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPGetIterationNumber.html")
@wrapper(:KSPView,PetscErrorCode,(KSP,PetscViewer),(ksp,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPView.html")
@wrapper(:KSPSetInitialGuessNonzero,PetscErrorCode,(KSP,PetscBool),(ksp,flg),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetInitialGuessNonzero.html")
@wrapper(:KSPSetType,PetscErrorCode,(KSP,KSPType),(ksp,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetType.html")
@wrapper(:KSPGetPC,PetscErrorCode,(KSP,Ptr{PC}),(ksp,pc),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPGetPC.html")
@wrapper(:PCSetType,PetscErrorCode,(PC,PCType),(pc,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCSetType.html")

end # module
