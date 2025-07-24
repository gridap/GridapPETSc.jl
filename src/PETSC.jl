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

@wrapper(:PetscInitializeNoArguments,PetscErrorCode,(),(),"https://petsc.org/release/manualpages/Sys/PetscInitializeNoArguments/")
@wrapper(:PetscInitializeNoPointers,PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),(argc,args,filename,help),"")
@wrapper(:PetscFinalize,PetscErrorCode,(),(),"https://petsc.org/release/manualpages/Sys/PetscFinalize/")
@wrapper(:PetscFinalized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://petsc.org/release/manualpages/Sys/PetscFinalized/")
@wrapper(:PetscInitialized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"https://petsc.org/release/manualpages/Sys/PetscInitialized/")

# viewer related functions

"""
Julia alias for `PetscViewer` C type.

See [PETSc manual](https://petsc.org/release/manualpages/Viewer/PetscViewer/).
"""
struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PetscViewer) = v.ptr

@wrapper(:PETSC_VIEWER_STDOUT_,PetscViewer,(MPI.Comm,),(comm,),"https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_STDOUT_/")
@wrapper(:PETSC_VIEWER_DRAW_,PetscViewer,(MPI.Comm,),(comm,),"https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_DRAW_/")

"""
    @PETSC_VIEWER_STDOUT_SELF

See [PETSc manual](https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_STDOUT_SELF/).
"""
macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end

"""
    @PETSC_VIEWER_STDOUT_WORLD

See [PETSc manual](https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_STDOUT_WORLD/).
"""
macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end

"""
    @PETSC_VIEWER_DRAW_SELF

See [PETSc manual](https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_DRAW_SELF/).
"""
macro PETSC_VIEWER_DRAW_SELF()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_SELF)
  end
end

"""
    @PETSC_VIEWER_DRAW_WORLD

See [PETSc manual](https://petsc.org/release/manualpages/Viewer/PETSC_VIEWER_DRAW_WORLD/).
"""
macro PETSC_VIEWER_DRAW_WORLD()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_WORLD)
  end
end

# Vector related functions

"""
Julia alias for the `InsertMode` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Sys/InsertMode/).
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
Julia alias for the `PetscCopyMode` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Sys/PetscCopyMode/).
"""
@enum PetscCopyMode begin
  PETSC_COPY_VALUES
  PETSC_OWN_POINTER
  PETSC_USE_POINTER
end

"""
Julia alias for the `VecOption` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Vec/VecSetOption/).
"""
@enum VecOption begin
  VEC_IGNORE_OFF_PROC_ENTRIES
  VEC_IGNORE_NEGATIVE_INDICES
  VEC_SUBSET_OFF_PROC_ENTRIES
end

"""
Julia alias for the `NormType` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Vec/NormType/).
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

See [PETSc manual](https://petsc.org/release/manualpages/Vec/Vec/).
"""
struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Vec) = v.ptr

@wrapper(:VecCreateSeq,PetscErrorCode,(MPI.Comm,PetscInt,Ptr{Vec}),(comm,n,vec),"https://petsc.org/release/manualpages/Vec/VecCreateSeq/")
@wrapper(:VecCreateSeqWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,array,vec),"https://petsc.org/release/manualpages/Vec/VecCreateSeqWithArray/")
@wrapper(:VecCreateGhost,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Vec}),(comm,n,N,nghost,ghosts,vv),"https://petsc.org/release/manualpages/Vec/VecCreateGhost/")
@wrapper(:VecCreateGhostWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},Ptr{Vec}),(comm,n,N,nghost,ghosts,array,vv),"https://petsc.org/release/manualpages/Vec/VecCreateGhostWithArray/")
@wrapper(:VecCreateMPI,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Vec),(comm,n,N,v),"https://petsc.org/release/manualpages/Vec/VecCreateMPI/")
@wrapper(:VecDestroy,PetscErrorCode,(Ptr{Vec},),(vec,),"https://petsc.org/release/manualpages/Vec/VecDestroy/")
@wrapper(:VecView,PetscErrorCode,(Vec,PetscViewer),(vec,viewer),"https://petsc.org/release/manualpages/Vec/VecView/")
@wrapper(:VecSetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(x,ni,ix,y,iora),"https://petsc.org/release/manualpages/Vec/VecSetValues/")
@wrapper(:VecGetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(x,ni,ix,y),"https://petsc.org/release/manualpages/Vec/VecGetValues/")
@wrapper(:VecGetArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecGetArray/")
@wrapper(:VecGetArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecGetArrayRead/")
@wrapper(:VecGetArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecGetArrayWrite/")
@wrapper(:VecRestoreArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecRestoreArray/")
@wrapper(:VecRestoreArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecRestoreArrayRead/")
@wrapper(:VecRestoreArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"https://petsc.org/release/manualpages/Vec/VecRestoreArrayWrite/")
@wrapper(:VecGetSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"https://petsc.org/release/manualpages/Vec/VecGetSize/")
@wrapper(:VecGetLocalSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"https://petsc.org/release/manualpages/Vec/VecGetLocalSize/")
@wrapper(:VecAssemblyBegin,PetscErrorCode,(Vec,),(vec,),"https://petsc.org/release/manualpages/Vec/VecAssemblyBegin/")
@wrapper(:VecAssemblyEnd,PetscErrorCode,(Vec,),(vec,),"https://petsc.org/release/manualpages/Vec/VecAssemblyEnd/")
@wrapper(:VecPlaceArray,PetscErrorCode,(Vec,Ptr{PetscScalar}),(vec,array),"https://petsc.org/release/manualpages/Vec/VecPlaceArray/")
@wrapper(:VecResetArray,PetscErrorCode,(Vec,),(vec,),"https://petsc.org/release/manualpages/Vec/VecResetArray/")
@wrapper(:VecScale,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"https://petsc.org/release/manualpages/Vec/VecScale/")
@wrapper(:VecSet,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"https://petsc.org/release/manualpages/Vec/VecSet/")
@wrapper(:VecDuplicate,PetscErrorCode,(Vec,Ptr{Vec}),(v,newv),"https://petsc.org/release/manualpages/Vec/VecDuplicate/")
@wrapper(:VecCopy,PetscErrorCode,(Vec,Vec),(x,y),"https://petsc.org/release/manualpages/Vec/VecCopy/")
@wrapper(:VecAXPY,PetscErrorCode,(Vec,PetscScalar,Vec),(y,alpha,x),"https://petsc.org/release/manualpages/Vec/VecAXPY/")
@wrapper(:VecAYPX,PetscErrorCode,(Vec,PetscScalar,Vec),(y,beta,x),"https://petsc.org/release/manualpages/Vec/VecAYPX/")
@wrapper(:VecAXPBY,PetscErrorCode,(Vec,PetscScalar,PetscScalar,Vec),(y,alpha,beta,x),"https://petsc.org/release/manualpages/Vec/VecAXPBY/")
@wrapper(:VecSetOption,PetscErrorCode,(Vec,VecOption,PetscBool),(x,op,flg),"https://petsc.org/release/manualpages/Vec/VecSetOption/")
@wrapper(:VecNorm,PetscErrorCode,(Vec,NormType,Ptr{PetscReal}),(x,typ,val),"https://petsc.org/release/manualpages/Vec/VecNorm/")
@wrapper(:VecGhostGetLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"https://petsc.org/release/manualpages/Vec/VecGhostGetLocalForm/")
@wrapper(:VecGhostRestoreLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"https://petsc.org/release/manualpages/Vec/VecGhostRestoreLocalForm/")
@wrapper(:VecSetBlockSize,PetscErrorCode,(Vec,PetscInt),(v,bs),"https://petsc.org/release/manualpages/Vec/VecSetBlockSize/")

# Matrix related functions

"""
Julia alias for the `MatAssemblyType` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatAssemblyType/).
"""
@enum MatAssemblyType begin
  MAT_FINAL_ASSEMBLY=0
  MAT_FLUSH_ASSEMBLY=1
end

"""
Julia alias for the `MatDuplicateOption` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatDuplicateOption/).
"""
@enum MatDuplicateOption begin
  MAT_DO_NOT_COPY_VALUES
  MAT_COPY_VALUES
  MAT_SHARE_NONZERO_PATTERN
end

"""
Julia alias for the `MatReuse` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatReuse/).
"""
@enum MatReuse begin
  MAT_INITIAL_MATRIX
  MAT_REUSE_MATRIX
  MAT_IGNORE_MATRIX
  MAT_INPLACE_MATRIX
end

"""
Julia alias for the `MatInfoType` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatInfoType/).
"""
@enum MatInfoType begin
  MAT_LOCAL=1
  MAT_GLOBAL_MAX=2
  MAT_GLOBAL_SUM=3
end

"""
Julia alias for the `MatStructure` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatStructure/).
"""
@enum MatStructure begin
  DIFFERENT_NONZERO_PATTERN
  SUBSET_NONZERO_PATTERN
  SAME_NONZERO_PATTERN
  UNKNOWN_NONZERO_PATTERN
end

"""
Julia alias to `PetscLogDouble` C type.

See [PETSc manual](https://petsc.org/release/manualpages/Sys/PetscLogDouble/).
"""
const PetscLogDouble = Cdouble

"""
Julia alias for the `MatInfo` C struct.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatInfo/).
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

See [PETSc manual](https://petsc.org/release/manualpages/Sys/PETSC_DEFAULT/).
"""
const PETSC_DEFAULT = Cint(-2)

"""
Julia constant storing the `PETSC_DECIDE` value.

See [PETSc manual](https://petsc.org/release/manualpages/Sys/PETSC_DECIDE/).
"""
const PETSC_DECIDE = Cint(-1)

"""
Julia constant storing the `PETSC_DETERMINE` value.

See [PETSc manual](https://petsc.org/release/manualpages/Sys/PETSC_DETERMINE/).
"""
const PETSC_DETERMINE = PETSC_DECIDE

"""
Julia alias for `MatType` C type.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatType/).
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

See [PETSc manual](https://petsc.org/release/manualpages/Mat/Mat/).
"""
struct Mat
  ptr::Ptr{Cvoid}
end
Mat() = Mat(Ptr{Cvoid}())
Base.convert(::Type{Mat},p::Ptr{Cvoid}) = Mat(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Mat) = v.ptr

@wrapper(:MatCreateAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,M,N,d_nz,d_nnz,o_nz,o_nnz,mat),"https://petsc.org/release/manualpages/Mat/MatCreateAIJ/")
@wrapper(:MatCreateSeqAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,nz,nnz,mat),"https://petsc.org/release/manualpages/Mat/MatCreateSeqAIJ/")
@wrapper(:MatCreateSeqAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,i,j,a,mat),"https://petsc.org/release/manualpages/Mat/MatCreateSeqAIJWithArrays/")
@wrapper(:MatCreateMPIAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,M,N,i,j,a,mat),"https://petsc.org/release/manualpages/Mat/MatCreateMPIAIJWithArrays/")
@wrapper(:MatDestroy,PetscErrorCode,(Ptr{Mat},),(A,),"https://petsc.org/release/manualpages/Mat/MatDestroy/")
@wrapper(:MatView,PetscErrorCode,(Mat,PetscViewer),(mat,viewer),"https://petsc.org/release/manualpages/Mat/MatView/")
@wrapper(:MatSetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(mat,m,idxm,n,idxn,v,addv),"https://petsc.org/release/manualpages/Mat/MatSetValues/")
@wrapper(:MatGetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(mat,m,idxm,n,idxn,v),"https://petsc.org/release/manualpages/Mat/MatGetValues/")
@wrapper(:MatAssemblyBegin,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://petsc.org/release/manualpages/Mat/MatAssemblyBegin/")
@wrapper(:MatAssemblyEnd,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"https://petsc.org/release/manualpages/Mat/MatAssemblyEnd/")
@wrapper(:MatGetSize,PetscErrorCode,(Mat,Ptr{PetscInt},Ptr{PetscInt}),(mat,m,n),"https://petsc.org/release/manualpages/Mat/MatGetSize/")
@wrapper(:MatEqual,PetscErrorCode,(Mat,Mat,Ptr{PetscBool}),(A,B,flg),"https://petsc.org/release/manualpages/Mat/MatEqual/")
@wrapper(:MatMultAdd,PetscErrorCode,(Mat,Vec,Vec,Vec),(mat,v1,v2,v3),"https://petsc.org/release/manualpages/Mat/MatMultAdd/")
@wrapper(:MatMult,PetscErrorCode,(Mat,Vec,Vec),(mat,x,y),"https://petsc.org/release/manualpages/Mat/MatMult/")
@wrapper(:MatScale,PetscErrorCode,(Mat,PetscScalar),(mat,alpha),"https://petsc.org/release/manualpages/Mat/MatScale/")
@wrapper(:MatConvert,PetscErrorCode,(Mat,MatType,MatReuse,Ptr{Mat}),(mat,newtype,reuse,M),"https://petsc.org/release/manualpages/Mat/MatConvert/")
@wrapper(:MatGetInfo,PetscErrorCode,(Mat,MatInfoType,Ptr{MatInfo}),(mat,flag,info),"https://petsc.org/release/manualpages/Mat/MatGetInfo/")
@wrapper(:MatZeroEntries,PetscErrorCode,(Mat,),(mat,),"https://petsc.org/release/manualpages/Mat/MatZeroEntries/")
@wrapper(:MatCopy,PetscErrorCode,(Mat,Mat,MatStructure),(A,B,str),"https://petsc.org/release/manualpages/Mat/MatCopy/")
@wrapper(:MatSetBlockSize,PetscErrorCode,(Mat,PetscInt),(mat,bs),"https://petsc.org/release/manualpages/Mat/MatSetBlockSize/")
@wrapper(:MatMumpsSetIcntl,PetscErrorCode,(Mat,PetscInt,PetscInt),(mat,icntl,val),"https://petsc.org/release/manualpages/Mat/MatMumpsSetIcntl/")
@wrapper(:MatMumpsSetCntl,PetscErrorCode,(Mat,PetscInt,PetscReal),(mat,icntl,val),"https://petsc.org/release/manualpages/Mat/MatMumpsSetCntl/")

# Null space related

"""
Julia alias for the `MatNullSpace` C type.

See [PETSc manual](https://petsc.org/release/manualpages/Mat/MatNullSpace/).
"""
struct MatNullSpace
  ptr::Ptr{Cvoid}
end
MatNullSpace() = MatNullSpace(Ptr{Cvoid}())
Base.convert(::Type{MatNullSpace},p::Ptr{Cvoid}) = MatNullSpace(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::MatNullSpace) = v.ptr

@wrapper(:MatSetNullSpace,PetscErrorCode,(Mat,MatNullSpace),(mat,nullsp),"https://petsc.org/release/manualpages/Mat/MatSetNullSpace/")
@wrapper(:MatSetNearNullSpace,PetscErrorCode,(Mat,MatNullSpace),(mat,nullsp),"https://petsc.org/release/manualpages/Mat/MatSetNearNullSpace/")
@wrapper(:MatNullSpaceCreateRigidBody,PetscErrorCode,(Vec,Ptr{MatNullSpace}),(coords,sp),"https://petsc.org/release/manualpages/Mat/MatNullSpaceCreateRigidBody/")
@wrapper(:MatNullSpaceCreate,PetscErrorCode,(MPI.Comm,PetscBool,PetscInt,Ptr{Vec},Ptr{MatNullSpace}),(comm,has_cnst,n,vecs,sp),"https://petsc.org/release/manualpages/Mat/MatNullSpaceCreate/")
@wrapper(:MatNullSpaceDestroy,PetscErrorCode,(Ptr{MatNullSpace},),(ns,),"https://petsc.org/release/manualpages/Mat/MatNullSpaceDestroy/")

# KSP and PC related things

"""
Julia alias for `KSPType` C type.

See [PETSc manual](https://petsc.org/release/manualpages/KSP/KSPType/).
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

See [PETSc manual](https://petsc.org/release/manualpages/PC/PCType/).
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

See [PETSc manual](https://petsc.org/release/manualpages/KSP/KSP/).
"""
struct KSP
  ptr::Ptr{Cvoid}
end
KSP() = KSP(Ptr{Cvoid}())
Base.convert(::Type{KSP},p::Ptr{Cvoid}) = KSP(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::KSP) = v.ptr

"""
Julia alias for the `PC` C type.

See [PETSc manual](https://petsc.org/release/manualpages/PC/PC/).
"""
struct PC
  ptr::Ptr{Cvoid}
end
PC() = PC(Ptr{Cvoid}())
Base.convert(::Type{PC},p::Ptr{Cvoid}) = PC(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PC) = v.ptr

@wrapper(:KSPCreate,PetscErrorCode,(MPI.Comm,Ptr{KSP}),(comm,inksp),"https://petsc.org/release/manualpages/KSP/KSPCreate/")
@wrapper(:KSPDestroy,PetscErrorCode,(Ptr{KSP},),(ksp,),"https://petsc.org/release/manualpages/KSP/KSPDestroy/")
@wrapper(:KSPSetFromOptions,PetscErrorCode,(KSP,),(ksp,),"https://petsc.org/release/manualpages/KSP/KSPSetFromOptions/")
@wrapper(:KSPSetOptionsPrefix,PetscErrorCode,(KSP,Cstring),(ksp,prefix),"https://petsc.org/release/manualpages/KSP/KSPSetOptionsPrefix/")
@wrapper(:KSPSetUp,PetscErrorCode,(KSP,),(ksp,),"https://petsc.org/release/manualpages/KSP/KSPSetUp/")
@wrapper(:KSPSetOperators,PetscErrorCode,(KSP,Mat,Mat),(ksp,Amat,Pmat),"https://petsc.org/release/manualpages/KSP/KSPSetOperators/")
@wrapper(:KSPSetTolerances,PetscErrorCode,(KSP,PetscReal,PetscReal,PetscReal,PetscInt),(ksp,rtol,abstol,dtol,maxits),"https://petsc.org/release/manualpages/KSP/KSPSetTolerances/")
@wrapper(:KSPGetOperators,PetscErrorCode,(KSP,Ptr{Mat},Ptr{Mat}),(ksp,Amat,Pmat),"https://petsc.org/release/manualpages/KSP/KSPGetOperators/")
@wrapper(:KSPSolve,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"https://petsc.org/release/manualpages/KSP/KSPSolve/")
@wrapper(:KSPSolveTranspose,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"https://petsc.org/release/manualpages/KSP/KSPSolveTranspose/")
@wrapper(:KSPGetIterationNumber,PetscErrorCode,(KSP,Ptr{PetscInt}),(ksp,its),"https://petsc.org/release/manualpages/KSP/KSPGetIterationNumber/")
@wrapper(:KSPGetResidualNorm,PetscErrorCode,(KSP,Ptr{PetscReal}),(ksp,rnorm),"https://petsc.org/release/manualpages/KSP/KSPGetResidualNorm/")
@wrapper(:KSPView,PetscErrorCode,(KSP,PetscViewer),(ksp,viewer),"https://petsc.org/release/manualpages/KSP/KSPView/")
@wrapper(:KSPSetInitialGuessNonzero,PetscErrorCode,(KSP,PetscBool),(ksp,flg),"https://petsc.org/release/manualpages/KSP/KSPSetInitialGuessNonzero/")
@wrapper(:KSPSetType,PetscErrorCode,(KSP,KSPType),(ksp,typ),"https://petsc.org/release/manualpages/KSP/KSPSetType/")
@wrapper(:KSPGetPC,PetscErrorCode,(KSP,Ptr{PC}),(ksp,pc),"https://petsc.org/release/manualpages/KSP/KSPGetPC/")

@wrapper(:PCSetFromOptions,PetscErrorCode,(PC,),(pc,),"https://petsc.org/release/manualpages/PC/PCSetFromOptions/")
@wrapper(:PCSetType,PetscErrorCode,(PC,PCType),(pc,typ),"https://petsc.org/release/manualpages/PC/PCSetType/")
@wrapper(:PCView,PetscErrorCode,(PC,PetscViewer),(pc,viewer),"https://petsc.org/release/manualpages/PC/PCView/")
@wrapper(:PCFactorSetMatSolverType,PetscErrorCode,(PC,PCType),(pc,typ),"https://petsc.org/release/manualpages/PC/PCFactorSetMatSolverType/")
@wrapper(:PCFactorSetUpMatSolverType,PetscErrorCode,(PC,),(pc,),"https://petsc.org/release/manualpages/PC/PCFactorSetUpMatSolverType/")
@wrapper(:PCFactorGetMatrix,PetscErrorCode,(PC,Ptr{Mat}),(ksp,mat),"https://petsc.org/release/manualpages/PC/PCFactorGetMatrix/")
@wrapper(:PCSetCoordinates,PetscErrorCode,(PC,PetscInt,PetscInt,Ptr{PetscScalar}),(pc,dim,nloc,coords),"https://petsc.org/release/manualpages/PC/PCSetCoordinates/")
@wrapper(:PCSetOperators,PetscErrorCode,(PC,Mat,Mat),(pc,Amat,Pmat),"https://petsc.org/release/manualpages/PC/PCSetOperators/")
@wrapper(:PCGetOperators,PetscErrorCode,(PC,Ptr{Mat},Ptr{Mat}),(pc,Amat,Pmat),"https://petsc.org/release/manualpages/PC/PCGetOperators/")

"""
Julia alias for the `SNES` C type.

See [PETSc manual](https://petsc.org/release/manualpages/SNES/SNES/).
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

"""
Julia alias to `SNESConvergedReason` C enum.

See [PETSc manual](https://petsc.org/release/manualpages/SNES/SNESConvergedReason/).
"""
@enum SNESConvergedReason begin
  SNES_CONVERGED_FNORM_ABS      = 2         # ||F|| < atol
  SNES_CONVERGED_FNORM_RELATIVE = 3         # ||F|| < rtol*||F_initial||
  SNES_CONVERGED_SNORM_RELATIVE = 4         # Newton computed step size small; || delta x || < stol || x ||
  SNES_CONVERGED_ITS            = 5         # maximum iterations reached
  SNES_BREAKOUT_INNER_ITER      = 6         # Flag to break out of inner loop after checking custom convergence.
                                            # it is used in multi-phase flow when state changes diverged
  SNES_DIVERGED_FUNCTION_DOMAIN      = -1   # the new x location passed the function is not in the domain of F
  SNES_DIVERGED_FUNCTION_COUNT       = -2
  SNES_DIVERGED_LINEAR_SOLVE         = -3   # the linear solve failed
  SNES_DIVERGED_FNORM_NAN            = -4
  SNES_DIVERGED_MAX_IT               = -5
  SNES_DIVERGED_LINE_SEARCH          = -6   # the line search failed
  SNES_DIVERGED_INNER                = -7   # inner solve failed
  SNES_DIVERGED_LOCAL_MIN            = -8   # || J^T b || is small, implies converged to local minimum of F()
  SNES_DIVERGED_DTOL                 = -9   # || F || > divtol*||F_initial||
  SNES_DIVERGED_JACOBIAN_DOMAIN      = -10  # Jacobian calculation does not make sense
  SNES_DIVERGED_TR_DELTA             = -11

  SNES_CONVERGED_ITERATING = 0
end

@wrapper(:SNESCreate,PetscErrorCode,(MPI.Comm,Ptr{SNES}),(comm,snes),"https://petsc.org/release/manualpages/SNES/SNESCreate/")
@wrapper(:SNESSetFunction,PetscErrorCode,(SNES,Vec,Ptr{Cvoid},Ptr{Cvoid}),(snes,vec,fptr,ctx),"https://petsc.org/release/manualpages/SNES/SNESSetFunction/")
@wrapper(:SNESSetJacobian,PetscErrorCode,(SNES,Mat,Mat,Ptr{Cvoid},Ptr{Cvoid}),(snes,A,P,jacptr,ctx),"https://petsc.org/release/manualpages/SNES/SNESSetJacobian/")
@wrapper(:SNESSolve,PetscErrorCode,(SNES,Vec,Vec),(snes,b,x),"https://petsc.org/release/manualpages/SNES/SNESSolve/")
@wrapper(:SNESDestroy,PetscErrorCode,(Ptr{SNES},),(snes,),"https://petsc.org/release/manualpages/SNES/SNESDestroy/")
@wrapper(:SNESSetFromOptions,PetscErrorCode,(SNES,),(snes,),"https://petsc.org/release/manualpages/SNES/SNESSetFromOptions/")
@wrapper(:SNESView,PetscErrorCode,(SNES,PetscViewer),(snes,viewer),"https://petsc.org/release/manualpages/SNES/SNESView/")
@wrapper(:SNESSetType,PetscErrorCode,(SNES,SNESType),(snes,type),"https://petsc.org/release/manualpages/SNES/SNESSetType/")
@wrapper(:SNESGetKSP,PetscErrorCode,(SNES,Ptr{KSP}),(snes,ksp),"https://petsc.org/release/manualpages/SNES/SNESGetKSP/")
@wrapper(:SNESGetIterationNumber,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"https://petsc.org/release/manualpages/SNES/SNESGetIterationNumber/")
@wrapper(:SNESGetLinearSolveIterations,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"https://petsc.org/release/manualpages/SNES/SNESGetLinearSolveIterations/")
@wrapper(:SNESSetCountersReset,PetscErrorCode,(SNES,PetscBool),(snes,reset),"https://petsc.org/release/manualpages/SNES/SNESSetCountersReset/")
@wrapper(:SNESGetNumberFunctionEvals,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfuncs),"https://petsc.org/release/manualpages/SNES/SNESGetNumberFunctionEvals/")
@wrapper(:SNESGetLinearSolveFailures,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfails),"https://petsc.org/release/manualpages/SNES/SNESGetLinearSolveFailures/")
@wrapper(:SNESSetConvergenceTest,PetscErrorCode,(SNES,Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),(snes,convtest,cctx,destroy),"https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/")
@wrapper(:SNESConvergedDefault,PetscErrorCode,(SNES,PetscInt,PetscReal,PetscReal,PetscReal,Ptr{SNESConvergedReason},Ptr{Cvoid}),(snes,it,xnorm,gnorm,f,reason,user),"https://petsc.org/release/manualpages/SNES/SNESConvergedDefault/")

# Garbage collection of PETSc objects

@wrapper(:PetscObjectRegisterDestroy,PetscErrorCode,(Ptr{Cvoid},),(obj,),"https://petsc.org/release/manualpages/Sys/PetscObjectRegisterDestroy/")
@wrapper(:PetscObjectRegisterDestroyAll,PetscErrorCode,(),(),"https://petsc.org/release/manualpages/Sys/PetscObjectRegisterDestroyAll/")

# IS - Index sets

"""
Julia alias for the `IS` C type.

See [PETSc manual](https://petsc.org/release/manualpages/IS/IS/).
"""
struct IS
  ptr::Ptr{Cvoid}
end

const ISType = Cstring
const ISGENERAL = "general"
const ISSTRIDE  = "stride"
const ISBLOCK   = "block"

@wrapper(:ISCreateGeneral,PetscErrorCode,(MPI.Comm,PetscInt,Ptr{PetscInt},PetscCopyMode,Ptr{IS}),(comm,n,idx,mode,is),"https://petsc.org/release/manualpages/IS/ISCreateGeneral/")
@wrapper(:ISCreateStride,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{IS}),(comm,n,first,step,is),"https://petsc.org/release/manualpages/IS/ISCreateStride/")
@wrapper(:ISCreateBlock,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},PetscCopyMode,Ptr{IS}),(comm,bs,n,idx,mode,is),"https://petsc.org/release/manualpages/IS/ISCreateBlock/")
@wrapper(:ISSetType,PetscErrorCode,(IS,ISType),(is,method),"https://petsc.org/release/manualpages/IS/ISSetType/")
@wrapper(:ISDuplicate,PetscErrorCode,(IS,Ptr{IS}),(is,newis),"https://petsc.org/release/manualpages/IS/ISDuplicate/")
@wrapper(:ISGetSize,PetscErrorCode,(IS,Ptr{PetscInt}),(is,n),"https://petsc.org/release/manualpages/IS/ISGetSize/")
@wrapper(:ISGetIndices,PetscErrorCode,(IS,Ptr{Ptr{PetscInt}}),(is,ptr),"https://petsc.org/release/manualpages/IS/ISGetIndices/")
@wrapper(:ISGeneralSetIndices,PetscErrorCode,(IS,PetscInt,Ptr{PetscInt},PetscCopyMode),(is,n,idx,mode),"https://petsc.org/release/manualpages/IS/ISGeneralSetIndices/")
@wrapper(:ISBlockSetIndices,PetscErrorCode,(IS,PetscInt,PetscInt,Ptr{PetscInt},PetscCopyMode),(is,bs,n,idx,mode),"https://petsc.org/release/manualpages/IS/ISBlockSetIndices/")
@wrapper(:ISDestroy,PetscErrorCode,(Ptr{IS},),(is,),"https://petsc.org/release/manualpages/IS/ISDestroy/")

# HYPRE

@wrapper(:PCHYPRESetDiscreteGradient,PetscErrorCode,(PC,Mat),(pc,G),"https://petsc.org/release/manualpages/PC/PCHYPRESetDiscreteGradient/")
@wrapper(:PCHYPRESetDiscreteCurl,PetscErrorCode,(PC,Mat),(pc,C),"https://petsc.org/release/manualpages/PC/PCHYPRESetDiscreteCurl/")
@wrapper(:PCHYPRESetAlphaPoissonMatrix,PetscErrorCode,(PC,Mat),(pc,A),"https://petsc.org/release/manualpages/PC/PCHYPRESetAlphaPoissonMatrix/")
@wrapper(:PCHYPRESetBetaPoissonMatrix,PetscErrorCode,(PC,Mat),(pc,A),"https://petsc.org/release/manualpages/PC/PCHYPRESetBetaPoissonMatrix/")
@wrapper(:PCHYPRESetInterpolations,PetscErrorCode,(PC,PetscInt,Mat,Ptr{Mat},Mat,Ptr{Mat}),(pc,dim,RT_PiFull,RT_Pi,ND_PiFull,ND_Pi),"https://petsc.org/release/manualpages/PC/PCHYPRESetInterpolations/")
@wrapper(:PCHYPRESetEdgeConstantVectors,PetscErrorCode,(PC,Vec,Vec,Vec),(pc,ozz,zoz,zzo),"https://petsc.org/release/manualpages/PC/PCHYPRESetEdgeConstantVectors/")
@wrapper(:PCHYPREAMSSetInteriorNodes,PetscErrorCode,(PC,Vec),(pc,interior),"https://petsc.org/release/manualpages/PC/PCHYPREAMSSetInteriorNodes/")
end # module
