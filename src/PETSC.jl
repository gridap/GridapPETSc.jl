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
using GridapPETSc: _PRELOADS
using Gridap.Helpers: @check
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
  hn = Symbol("$(fn.value)_handle")
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
    const $hn = Ref(C_NULL)
    push!(_PRELOADS,($hn,$fn))
    @doc $str
    @inline function $(fn.value)($(args.args...))
      @check $(hn)[] != C_NULL "Missing symbol. Re-configure and compile PETSc."
      ccall($(hn)[],$rt,$argts,$(args.args...))
    end
  end
  esc(expr)
end

#Petsc init related functions

@wrapper(:PetscInitializeNoArguments,PetscErrorCode,(),(),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscInitializeNoArguments.html")
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
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PetscViewer) = v.ptr

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
Julia alias for the `VecOption` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetOption.html).
"""
@enum VecOption begin
  VEC_IGNORE_OFF_PROC_ENTRIES
  VEC_IGNORE_NEGATIVE_INDICES
  VEC_SUBSET_OFF_PROC_ENTRIES
end

"""
Julia alias for the `NormType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/NormType.html).
"""
@enum NormType begin
  NORM_1=0
  NORM_2=1
  NORM_FROBENIUS=2
  NORM_INFINITY=3
  NORM_1_AND_2=4
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
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Vec) = v.ptr

@wrapper(:VecCreateSeq,PetscErrorCode,(MPI.Comm,PetscInt,Ptr{Vec}),(comm,n,vec),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeq.html")
@wrapper(:VecCreateSeqWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,array,vec),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCreateSeqWithArray.html")
@wrapper(:VecCreateGhost,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Vec}),(comm,n,N,nghost,ghosts,vv),"https://petsc.org/release/docs/manualpages/Vec/VecCreateGhost.html")
@wrapper(:VecCreateGhostWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},Ptr{Vec}),(comm,n,N,nghost,ghosts,array,vv),"https://petsc.org/release/docs/manualpages/Vec/VecCreateGhostWithArray.html")
@wrapper(:VecCreateMPI,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Vec),(comm,n,N,v),"https://petsc.org/release/docs/manualpages/Vec/VecCreateMPI.html")
@wrapper(:VecDestroy,PetscErrorCode,(Ptr{Vec},),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDestroy.html")
@wrapper(:VecView,PetscErrorCode,(Vec,PetscViewer),(vec,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecView.html")
@wrapper(:VecSetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(x,ni,ix,y,iora),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetValues.html")
@wrapper(:VecGetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(x,ni,ix,y),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetValues.html")
@wrapper(:VecGetArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecGetArray.html")
@wrapper(:VecGetArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecGetArrayRead.html")
@wrapper(:VecGetArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecGetArrayWrite.html")
@wrapper(:VecRestoreArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecRestoreArray.html")
@wrapper(:VecRestoreArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecRestoreArrayRead.html")
@wrapper(:VecRestoreArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/docs/manualpages/Vec/VecRestoreArrayWrite.html")
@wrapper(:VecGetSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetSize.html")
@wrapper(:VecGetLocalSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"https://petsc.org/release/docs/manualpages/Vec/VecGetLocalSize.html")
@wrapper(:VecAssemblyBegin,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyBegin.html")
@wrapper(:VecAssemblyEnd,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAssemblyEnd.html")
@wrapper(:VecPlaceArray,PetscErrorCode,(Vec,Ptr{PetscScalar}),(vec,array),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecPlaceArray.html")
@wrapper(:VecResetArray,PetscErrorCode,(Vec,),(vec,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecResetArray.html")
@wrapper(:VecScale,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecScale.html")
@wrapper(:VecSet,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSet.html")
@wrapper(:VecDuplicate,PetscErrorCode,(Vec,Ptr{Vec}),(v,newv),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDuplicate.html")
@wrapper(:VecCopy,PetscErrorCode,(Vec,Vec),(x,y),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecCopy.html")
@wrapper(:VecAXPY,PetscErrorCode,(Vec,PetscScalar,Vec),(y,alpha,x),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAXPY.html")
@wrapper(:VecAYPX,PetscErrorCode,(Vec,PetscScalar,Vec),(y,beta,x),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAYPX.html")
@wrapper(:VecAXPBY,PetscErrorCode,(Vec,PetscScalar,PetscScalar,Vec),(y,alpha,beta,x),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecAXPBY.html")
@wrapper(:VecSetOption,PetscErrorCode,(Vec,VecOption,PetscBool),(x,op,flg),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecSetOption.html")
@wrapper(:VecNorm,PetscErrorCode,(Vec,NormType,Ptr{PetscReal}),(x,typ,val),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecNorm.html")
@wrapper(:VecGhostGetLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"https://petsc.org/release/docs/manualpages/Vec/VecGhostGetLocalForm.html")
@wrapper(:VecGhostRestoreLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"https://petsc.org/release/docs/manualpages/Vec/VecGhostRestoreLocalForm.html")
@wrapper(:VecSetBlockSize,PetscErrorCode,(Vec,PetscInt),(v,bs),"https://petsc.org/release/manualpages/Vec/VecSetBlockSize.html")

# Matrix related functions

"""
Julia alias for the `MatAssemblyType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyType.html).
"""
@enum MatAssemblyType begin
  MAT_FINAL_ASSEMBLY=0
  MAT_FLUSH_ASSEMBLY=1
end

"""
Julia alias for the `MatDuplicateOption` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDuplicateOption.html).
"""
@enum MatDuplicateOption begin
  MAT_DO_NOT_COPY_VALUES
  MAT_COPY_VALUES
  MAT_SHARE_NONZERO_PATTERN
end

"""
Julia alias for the `MatReuse` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatReuse.html).
"""
@enum MatReuse begin
  MAT_INITIAL_MATRIX
  MAT_REUSE_MATRIX
  MAT_IGNORE_MATRIX
  MAT_INPLACE_MATRIX
end

"""
Julia alias for the `MatInfoType` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatInfoType.html).
"""
@enum MatInfoType begin
  MAT_LOCAL=1
  MAT_GLOBAL_MAX=2
  MAT_GLOBAL_SUM=3
end

"""
Julia alias for the `MatStructure` C enum.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatStructure.html).
"""
@enum MatStructure begin
  DIFFERENT_NONZERO_PATTERN
  SUBSET_NONZERO_PATTERN
  SAME_NONZERO_PATTERN
  UNKNOWN_NONZERO_PATTERN
end

"""
Julia alias to `PetscLogDouble` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PetscLogDouble.html).
"""
const PetscLogDouble = Cdouble

"""
Julia alias for the `MatInfo` C struct.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatInfo.html).
"""
struct MatInfo
  block_size         ::PetscLogDouble
  nz_allocated       ::PetscLogDouble
  nz_used            ::PetscLogDouble
  nz_unneeded        ::PetscLogDouble
  memory             ::PetscLogDouble
  assemblies         ::PetscLogDouble
  mallocs            ::PetscLogDouble
  fill_ratio_given   ::PetscLogDouble
  fill_ratio_needed  ::PetscLogDouble
  factor_mallocs     ::PetscLogDouble
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
Julia alias for `MatType` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatType.html).
"""
const MatType = Cstring
const MATSAME            = "same"
const MATMAIJ            = "maij"
const MATSEQMAIJ         = "seqmaij"
const MATMPIMAIJ         = "mpimaij"
const MATKAIJ            = "kaij"
const MATSEQKAIJ         = "seqkaij"
const MATMPIKAIJ         = "mpikaij"
const MATIS              = "is"
const MATAIJ             = "aij"
const MATSEQAIJ          = "seqaij"
const MATMPIAIJ          = "mpiaij"
const MATAIJCRL          = "aijcrl"
const MATSEQAIJCRL       = "seqaijcrl"
const MATMPIAIJCRL       = "mpiaijcrl"
const MATAIJCUSPARSE     = "aijcusparse"
const MATSEQAIJCUSPARSE  = "seqaijcusparse"
const MATMPIAIJCUSPARSE  = "mpiaijcusparse"
const MATAIJKOKKOS       = "aijkokkos"
const MATSEQAIJKOKKOS    = "seqaijkokkos"
const MATMPIAIJKOKKOS    = "mpiaijkokkos"
const MATAIJVIENNACL     = "aijviennacl"
const MATSEQAIJVIENNACL  = "seqaijviennacl"
const MATMPIAIJVIENNACL  = "mpiaijviennacl"
const MATAIJPERM         = "aijperm"
const MATSEQAIJPERM      = "seqaijperm"
const MATMPIAIJPERM      = "mpiaijperm"
const MATAIJSELL         = "aijsell"
const MATSEQAIJSELL      = "seqaijsell"
const MATMPIAIJSELL      = "mpiaijsell"
const MATAIJMKL          = "aijmkl"
const MATSEQAIJMKL       = "seqaijmkl"
const MATMPIAIJMKL       = "mpiaijmkl"
const MATBAIJMKL         = "baijmkl"
const MATSEQBAIJMKL      = "seqbaijmkl"
const MATMPIBAIJMKL      = "mpibaijmkl"
const MATSHELL           = "shell"
const MATDENSE           = "dense"
const MATDENSECUDA       = "densecuda"
const MATSEQDENSE        = "seqdense"
const MATSEQDENSECUDA    = "seqdensecuda"
const MATMPIDENSE        = "mpidense"
const MATMPIDENSECUDA    = "mpidensecuda"
const MATELEMENTAL       = "elemental"
const MATSCALAPACK       = "scalapack"
const MATBAIJ            = "baij"
const MATSEQBAIJ         = "seqbaij"
const MATMPIBAIJ         = "mpibaij"
const MATMPIADJ          = "mpiadj"
const MATSBAIJ           = "sbaij"
const MATSEQSBAIJ        = "seqsbaij"
const MATMPISBAIJ        = "mpisbaij"
const MATMFFD            = "mffd"
const MATNORMAL          = "normal"
const MATNORMALHERMITIAN = "normalh"
const MATLRC             = "lrc"
const MATSCATTER         = "scatter"
const MATBLOCKMAT        = "blockmat"
const MATCOMPOSITE       = "composite"
const MATFFT             = "fft"
const MATFFTW            = "fftw"
const MATSEQCUFFT        = "seqcufft"
const MATTRANSPOSEMAT    = "transpose"
const MATSCHURCOMPLEMENT = "schurcomplement"
const MATPYTHON          = "python"
const MATHYPRE           = "hypre"
const MATHYPRESTRUCT     = "hyprestruct"
const MATHYPRESSTRUCT    = "hypresstruct"
const MATSUBMATRIX       = "submatrix"
const MATLOCALREF        = "localref"
const MATNEST            = "nest"
const MATPREALLOCATOR    = "preallocator"
const MATSELL            = "sell"
const MATSEQSELL         = "seqsell"
const MATMPISELL         = "mpisell"
const MATDUMMY           = "dummy"
const MATLMVM            = "lmvm"
const MATLMVMDFP         = "lmvmdfp"
const MATLMVMBFGS        = "lmvmbfgs"
const MATLMVMSR1         = "lmvmsr1"
const MATLMVMBROYDEN     = "lmvmbroyden"
const MATLMVMBADBROYDEN  = "lmvmbadbroyden"
const MATLMVMSYMBROYDEN  = "lmvmsymbroyden"
const MATLMVMSYMBADBROYDEN = "lmvmsymbadbroyden"
const MATLMVMDIAGBROYDEN   = "lmvmdiagbroyden"
const MATCONSTANTDIAGONAL  = "constantdiagonal"
const MATHARA              = "hara"

const MATSOLVERSUPERLU          = "superlu"
const MATSOLVERSUPERLU_DIST     = "superlu_dist"
const MATSOLVERSTRUMPACK        = "strumpack"
const MATSOLVERUMFPACK          = "umfpack"
const MATSOLVERCHOLMOD          = "cholmod"
const MATSOLVERKLU              = "klu"
const MATSOLVERSPARSEELEMENTAL  = "sparseelemental"
const MATSOLVERELEMENTAL        = "elemental"
const MATSOLVERESSL             = "essl"
const MATSOLVERLUSOL            = "lusol"
const MATSOLVERMUMPS            = "mumps"
const MATSOLVERMKL_PARDISO      = "mkl_pardiso"
const MATSOLVERMKL_CPARDISO     = "mkl_cpardiso"
const MATSOLVERPASTIX           = "pastix"
const MATSOLVERMATLAB           = "matlab"
const MATSOLVERPETSC            = "petsc"
const MATSOLVERCUSPARSE         = "cusparse"

"""
Julia alias for the `Mat` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/Mat.html).
"""
struct Mat
  ptr::Ptr{Cvoid}
end
Mat() = Mat(Ptr{Cvoid}())
Base.convert(::Type{Mat},p::Ptr{Cvoid}) = Mat(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Mat) = v.ptr

@wrapper(:MatCreateAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,M,N,d_nz,d_nnz,o_nz,o_nnz,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateAIJ.html")
@wrapper(:MatCreateSeqAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,nz,nnz,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJ.html")
@wrapper(:MatCreateSeqAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,i,j,a,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCreateSeqAIJWithArrays.html")
@wrapper(:MatCreateMPIAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,M,N,i,j,a,mat),"https://petsc.org/release/docs/manualpages/Mat/MatCreateMPIAIJWithArrays.html")
@wrapper(:MatDestroy,PetscErrorCode,(Ptr{Mat},),(A,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatDestroy.html")
@wrapper(:MatView,PetscErrorCode,(Mat,PetscViewer),(mat,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatView.html")
@wrapper(:MatSetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(mat,m,idxm,n,idxn,v,addv),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValues.html")
@wrapper(:MatGetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(mat,m,idxm,n,idxn,v),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetValues.html")
@wrapper(:MatAssemblyBegin,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyBegin.html")
@wrapper(:MatAssemblyEnd,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatAssemblyEnd.html")
@wrapper(:MatGetSize,PetscErrorCode,(Mat,Ptr{PetscInt},Ptr{PetscInt}),(mat,m,n),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetSize.html")
@wrapper(:MatEqual,PetscErrorCode,(Mat,Mat,Ptr{PetscBool}),(A,B,flg),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatEqual.html")
@wrapper(:MatMultAdd,PetscErrorCode,(Mat,Vec,Vec,Vec),(mat,v1,v2,v3),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMultAdd.html")
@wrapper(:MatMult,PetscErrorCode,(Mat,Vec,Vec),(mat,x,y),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMult.html")
@wrapper(:MatScale,PetscErrorCode,(Mat,PetscScalar),(mat,alpha),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatScale.html")
@wrapper(:MatConvert,PetscErrorCode,(Mat,MatType,MatReuse,Ptr{Mat}),(mat,newtype,reuse,M),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatConvert.html")
@wrapper(:MatGetInfo,PetscErrorCode,(Mat,MatInfoType,Ptr{MatInfo}),(mat,flag,info),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetInfo.html")
@wrapper(:MatZeroEntries,PetscErrorCode,(Mat,),(mat,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatZeroEntries.html")
@wrapper(:MatCopy,PetscErrorCode,(Mat,Mat,MatStructure),(A,B,str),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatCopy.html")
@wrapper(:MatSetBlockSize,PetscErrorCode,(Mat,PetscInt),(mat,bs),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetBlockSize.html")
@wrapper(:MatMumpsSetIcntl,PetscErrorCode,(Mat,PetscInt,PetscInt),(mat,icntl,val),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMumpsSetIcntl.html")
@wrapper(:MatMumpsSetCntl,PetscErrorCode,(Mat,PetscInt,PetscReal),(mat,icntl,val),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMumpsSetCntl.html")

# Null space related

"""
Julia alias for the `MatNullSpace` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpace.html).
"""
struct MatNullSpace
  ptr::Ptr{Cvoid}
end
MatNullSpace() = MatNullSpace(Ptr{Cvoid}())
Base.convert(::Type{MatNullSpace},p::Ptr{Cvoid}) = MatNullSpace(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::MatNullSpace) = v.ptr

@wrapper(:MatSetNullSpace,PetscErrorCode,(Mat,MatNullSpace),(mat,nullsp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetNullSpace.html")
@wrapper(:MatSetNearNullSpace,PetscErrorCode,(Mat,MatNullSpace),(mat,nullsp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetNearNullSpace.html")
@wrapper(:MatNullSpaceCreateRigidBody,PetscErrorCode,(Vec,Ptr{MatNullSpace}),(coords,sp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpaceCreateRigidBody.html")
@wrapper(:MatNullSpaceCreate,PetscErrorCode,(MPI.Comm,PetscBool,PetscInt,Ptr{Vec},Ptr{MatNullSpace}),(comm,has_cnst,n,vecs,sp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpaceCreate.html")
@wrapper(:MatNullSpaceDestroy,PetscErrorCode,(Ptr{MatNullSpace},),(ns,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpaceDestroy.html")

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
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::KSP) = v.ptr

"""
Julia alias for the `PC` C type.

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PC.html).
"""
struct PC
  ptr::Ptr{Cvoid}
end
PC() = PC(Ptr{Cvoid}())
Base.convert(::Type{PC},p::Ptr{Cvoid}) = PC(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PC) = v.ptr

@wrapper(:KSPCreate,PetscErrorCode,(MPI.Comm,Ptr{KSP}),(comm,inksp),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPCreate.html")
@wrapper(:KSPDestroy,PetscErrorCode,(Ptr{KSP},),(ksp,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPDestroy.html")
@wrapper(:KSPSetFromOptions,PetscErrorCode,(KSP,),(ksp,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPSetFromOptions.html")
@wrapper(:KSPSetOptionsPrefix,PetscErrorCode,(KSP,Cstring),(ksp,prefix),"https://petsc.org/release/docs/manualpages/KSP/KSPSetOptionsPrefix.html")
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
@wrapper(:PCView,PetscErrorCode,(PC,PetscViewer),(pc,viewer),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCView.html")
@wrapper(:PCFactorSetMatSolverType,PetscErrorCode,(PC,PCType),(pc,typ),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCFactorSetMatSolverType.html")
@wrapper(:PCFactorSetUpMatSolverType,PetscErrorCode,(PC,),(pc,),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCFactorSetUpMatSolverType.html")
@wrapper(:PCFactorGetMatrix,PetscErrorCode,(PC,Ptr{Mat}),(ksp,mat),"https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCFactorGetMatrix.html")
@wrapper(:PCSetCoordinates,PetscErrorCode,(PC,PetscInt,PetscInt,Ptr{PetscScalar}),(pc,dim,nloc,coords),"https://petsc.org/release/manualpages/PC/PCSetCoordinates.html")

"""
Julia alias for the `SNES` C type.

See [PETSc manual](https://petsc.org/release/docs/manualpages/SNES/SNES.html).
"""
struct SNES
  ptr::Ptr{Cvoid}
end
SNES() = SNES(Ptr{Cvoid}())
Base.convert(::Type{SNES},p::Ptr{Cvoid}) = SNES(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::SNES) = v.ptr

const SNESType = Cstring
const SNESNEWTONLS         = "newtonls"
const SNESNEWTONTR         = "newtontr"
const SNESPYTHON           = "python"
const SNESNRICHARDSON      = "nrichardson"
const SNESKSPONLY          = "ksponly"
const SNESKSPTRANSPOSEONLY = "ksptransposeonly"
const SNESVINEWTONRSLS     = "vinewtonrsls"
const SNESVINEWTONSSLS     = "vinewtonssls"
const SNESNGMRES           = "ngmres"
const SNESQN               = "qn"
const SNESSHELL            = "shell"
const SNESNGS              = "ngs"
const SNESNCG              = "ncg"
const SNESFAS              = "fas"
const SNESMS               = "ms"
const SNESNASM             = "nasm"
const SNESANDERSON         = "anderson"
const SNESASPIN            = "aspin"
const SNESCOMPOSITE        = "composite"
const SNESPATCH            = "patch"


@wrapper(:SNESCreate,PetscErrorCode,(MPI.Comm,Ptr{SNES}),(comm,snes),"https://petsc.org/release/docs/manualpages/SNES/SNESCreate.html")
@wrapper(:SNESSetFunction,PetscErrorCode,(SNES,Vec,Ptr{Cvoid},Ptr{Cvoid}),(snes,vec,fptr,ctx),"https://petsc.org/release/docs/manualpages/SNES/SNESSetFunction.html")
@wrapper(:SNESSetJacobian,PetscErrorCode,(SNES,Mat,Mat,Ptr{Cvoid},Ptr{Cvoid}),(snes,A,P,jacptr,ctx),"https://petsc.org/release/docs/manualpages/SNES/SNESSetJacobian.html")
@wrapper(:SNESSolve,PetscErrorCode,(SNES,Vec,Vec),(snes,b,x),"https://petsc.org/release/docs/manualpages/SNES/SNESSolve.html")
@wrapper(:SNESDestroy,PetscErrorCode,(Ptr{SNES},),(snes,),"https://petsc.org/release/docs/manualpages/SNES/SNESDestroy.html")
@wrapper(:SNESSetFromOptions,PetscErrorCode,(SNES,),(snes,),"https://petsc.org/release/docs/manualpages/SNES/SNESSetFromOptions.html")
@wrapper(:SNESView,PetscErrorCode,(SNES,PetscViewer),(snes,viewer),"https://petsc.org/release/docs/manualpages/SNES/SNESView.html")
@wrapper(:SNESSetType,PetscErrorCode,(SNES,SNESType),(snes,type),"https://petsc.org/release/docs/manualpages/SNES/SNESSetType.html")
@wrapper(:SNESGetKSP,PetscErrorCode,(SNES,Ptr{KSP}),(snes,ksp),"https://petsc.org/release/docs/manualpages/SNES/SNESGetKSP.html")
@wrapper(:SNESGetIterationNumber,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"https://petsc.org/release/docs/manualpages/SNES/SNESGetIterationNumber.html")
@wrapper(:SNESGetLinearSolveIterations,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"https://petsc.org/release/docs/manualpages/SNES/SNESGetLinearSolveIterations.html")
@wrapper(:SNESSetCountersReset,PetscErrorCode,(SNES,PetscBool),(snes,reset),"https://petsc.org/release/docs/manualpages/SNES/SNESSetCountersReset.html")
@wrapper(:SNESGetNumberFunctionEvals,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfuncs),"https://petsc.org/release/docs/manualpages/SNES/SNESGetNumberFunctionEvals.html")
@wrapper(:SNESGetLinearSolveFailures,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfails),"https://petsc.org/release/docs/manualpages/SNES/SNESGetLinearSolveFailures.html")

# Garbage collection of PETSc objects
@wrapper(:PetscObjectRegisterDestroy,PetscErrorCode,(Ptr{Cvoid},),(obj,),"https://petsc.org/release/docs/manualpages/Sys/PetscObjectRegisterDestroy.html")
@wrapper(:PetscObjectRegisterDestroyAll,PetscErrorCode,(),(),"https://petsc.org/release/docs/manualpages/Sys/PetscObjectRegisterDestroyAll.html")

# HYPRE

@wrapper(:PCHYPRESetDiscreteGradient,PetscErrorCode,(PC,Mat),(pc,G),"https://petsc.org/release/manualpages/PC/PCHYPRESetDiscreteGradient.html")
@wrapper(:PCHYPRESetDiscreteCurl,PetscErrorCode,(PC,Mat),(pc,C),"https://petsc.org/release/manualpages/PC/PCHYPRESetDiscreteCurl.html")
@wrapper(:PCHYPRESetAlphaPoissonMatrix,PetscErrorCode,(PC,Mat),(pc,A),"https://petsc.org/release/manualpages/PC/PCHYPRESetAlphaPoissonMatrix.html")
@wrapper(:PCHYPRESetBetaPoissonMatrix,PetscErrorCode,(PC,Mat),(pc,A),"https://petsc.org/release/manualpages/PC/PCHYPRESetBetaPoissonMatrix.html")
@wrapper(:PCHYPRESetInterpolations,PetscErrorCode,(PC,PetscInt,Mat,Ptr{Mat},Mat,Ptr{Mat}),(pc,dim,RT_PiFull,RT_Pi,ND_PiFull,ND_Pi),"https://petsc.org/release/manualpages/PC/PCHYPRESetInterpolations.html")
@wrapper(:PCHYPRESetEdgeConstantVectors,PetscErrorCode,(PC,Vec,Vec,Vec),(pc,ozz,zoz,zzo),"https://petsc.org/release/manualpages/PC/PCHYPRESetEdgeConstantVectors.html")
@wrapper(:PCHYPREAMSSetInteriorNodes,PetscErrorCode,(PC,Vec),(pc,interior),"https://petsc.org/release/manualpages/PC/PCHYPREAMSSetInteriorNodes.html")
end # module
