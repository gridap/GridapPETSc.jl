var documenterSearchIndex = {"docs":
[{"location":"#GridapPETSc.jl","page":"Home","title":"GridapPETSc.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GridapPETSc,GridapPETSc.PETSC]","category":"page"},{"location":"#GridapPETSc.PETSC","page":"Home","title":"GridapPETSc.PETSC","text":"Low level interface with PETSC, which serve as the back-end in GridapPETSc.\n\nThe types and functions defined here are almost 1-to-1 to the corresponding C counterparts. In particular, the types defined can be directly used to call C PETSc routines via ccall. When a C function expects a pointer, use a Ref to the corresponding Julia alias. E.g., if an argument is PetscBool * in the C code, pass an object with type Ref{PetscBool} from the Julia code. Using this rule, PETSC.PetscInitialized can be called as\n\nflag = Ref{PetscBool}()\n@check_error_code PetscInitialized(flag)\nif flag[] == PETSC_TRUE\n  println(\"Petsc is initialized!\")\nend\n\n\n\n\n\n","category":"module"},{"location":"#GridapPETSc.PETSC.PETSC_DECIDE","page":"Home","title":"GridapPETSc.PETSC.PETSC_DECIDE","text":"Julia constant storing the PETSC_DECIDE value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"#GridapPETSc.PETSC.PETSC_DEFAULT","page":"Home","title":"GridapPETSc.PETSC.PETSC_DEFAULT","text":"Julia constant storing the PETSC_DEFAULT value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"#GridapPETSc.PETSC.PETSC_DETERMINE","page":"Home","title":"GridapPETSc.PETSC.PETSC_DETERMINE","text":"Julia constant storing the PETSC_DETERMINE value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"#GridapPETSc.PETSC.InsertMode","page":"Home","title":"GridapPETSc.PETSC.InsertMode","text":"Julia alias for the InsertMode C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.KSP","page":"Home","title":"GridapPETSc.PETSC.KSP","text":"Julia alias for the KSP C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.KSPType","page":"Home","title":"GridapPETSc.PETSC.KSPType","text":"Julia alias for KSPType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.Mat","page":"Home","title":"GridapPETSc.PETSC.Mat","text":"Julia alias for the Mat C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatAssemblyType","page":"Home","title":"GridapPETSc.PETSC.MatAssemblyType","text":"Julia alias for the MatAssemblyType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatDuplicateOption","page":"Home","title":"GridapPETSc.PETSC.MatDuplicateOption","text":"Julia alias for the MatDuplicateOption C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatInfo","page":"Home","title":"GridapPETSc.PETSC.MatInfo","text":"Julia alias for the MatInfo C struct.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatInfoType","page":"Home","title":"GridapPETSc.PETSC.MatInfoType","text":"Julia alias for the MatInfoType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatNullSpace","page":"Home","title":"GridapPETSc.PETSC.MatNullSpace","text":"Julia alias for the MatNullSpace C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatReuse","page":"Home","title":"GridapPETSc.PETSC.MatReuse","text":"Julia alias for the MatReuse C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatStructure","page":"Home","title":"GridapPETSc.PETSC.MatStructure","text":"Julia alias for the MatStructure C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.MatType","page":"Home","title":"GridapPETSc.PETSC.MatType","text":"Julia alias for MatType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.NormType","page":"Home","title":"GridapPETSc.PETSC.NormType","text":"Julia alias for the NormType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PC","page":"Home","title":"GridapPETSc.PETSC.PC","text":"Julia alias for the PC C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PCType","page":"Home","title":"GridapPETSc.PETSC.PCType","text":"Julia alias for PCType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscBool","page":"Home","title":"GridapPETSc.PETSC.PetscBool","text":"Julia alias to PetscBool C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscDataType","page":"Home","title":"GridapPETSc.PETSC.PetscDataType","text":"Julia alias to PetscDataType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscError","page":"Home","title":"GridapPETSc.PETSC.PetscError","text":"struct PetscError <: Exception\n  code::PetscErrorCode\nend\n\nCustom Exception thrown by @check_error_code.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscErrorCode","page":"Home","title":"GridapPETSc.PETSC.PetscErrorCode","text":"Julia alias to PetscErrorCode C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscInt","page":"Home","title":"GridapPETSc.PETSC.PetscInt","text":"Julia alias for PetscInt C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscLogDouble","page":"Home","title":"GridapPETSc.PETSC.PetscLogDouble","text":"Julia alias to PetscLogDouble C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscReal","page":"Home","title":"GridapPETSc.PETSC.PetscReal","text":"Julia alias for PetscReal C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscScalar","page":"Home","title":"GridapPETSc.PETSC.PetscScalar","text":"Julia alias for PetscScalar C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.PetscViewer","page":"Home","title":"GridapPETSc.PETSC.PetscViewer","text":"Julia alias for PetscViewer C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.SNES","page":"Home","title":"GridapPETSc.PETSC.SNES","text":"Julia alias for the SNES C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.Vec","page":"Home","title":"GridapPETSc.PETSC.Vec","text":"Julia alias for the Vec C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.VecOption","page":"Home","title":"GridapPETSc.PETSC.VecOption","text":"Julia alias for the VecOption C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"#GridapPETSc.PETSC.KSPCreate-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPCreate","text":"KSPCreate(comm, inksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.KSPDestroy","text":"KSPDestroy(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPGetIterationNumber-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPGetIterationNumber","text":"KSPGetIterationNumber(ksp, its)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPGetPC-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPGetPC","text":"KSPGetPC(ksp, pc)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetFromOptions-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetFromOptions","text":"KSPSetFromOptions(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetInitialGuessNonzero-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetInitialGuessNonzero","text":"KSPSetInitialGuessNonzero(ksp, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetOperators-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetOperators","text":"KSPSetOperators(ksp, Amat, Pmat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetOptionsPrefix-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetOptionsPrefix","text":"KSPSetOptionsPrefix(ksp, prefix)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetTolerances-NTuple{5, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetTolerances","text":"KSPSetTolerances(ksp, rtol, abstol, dtol, maxits)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetType-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetType","text":"KSPSetType(ksp, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSetUp-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSetUp","text":"KSPSetUp(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSolve-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSolve","text":"KSPSolve(ksp, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPSolveTranspose-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPSolveTranspose","text":"KSPSolveTranspose(ksp, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.KSPView-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.KSPView","text":"KSPView(ksp, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatAssemblyBegin-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatAssemblyBegin","text":"MatAssemblyBegin(mat, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatAssemblyEnd-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatAssemblyEnd","text":"MatAssemblyEnd(mat, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatConvert-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.MatConvert","text":"MatConvert(mat, newtype, reuse, M)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatCopy-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatCopy","text":"MatCopy(A, B, str)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatCreateAIJ-NTuple{10, Any}","page":"Home","title":"GridapPETSc.PETSC.MatCreateAIJ","text":"MatCreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatCreateMPIAIJWithArrays-NTuple{9, Any}","page":"Home","title":"GridapPETSc.PETSC.MatCreateMPIAIJWithArrays","text":"MatCreateMPIAIJWithArrays(comm, m, n, M, N, i, j, a, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatCreateSeqAIJ-NTuple{6, Any}","page":"Home","title":"GridapPETSc.PETSC.MatCreateSeqAIJ","text":"MatCreateSeqAIJ(comm, m, n, nz, nnz, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatCreateSeqAIJWithArrays-NTuple{7, Any}","page":"Home","title":"GridapPETSc.PETSC.MatCreateSeqAIJWithArrays","text":"MatCreateSeqAIJWithArrays(comm, m, n, i, j, a, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.MatDestroy","text":"MatDestroy(A)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatEqual-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatEqual","text":"MatEqual(A, B, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatGetInfo-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatGetInfo","text":"MatGetInfo(mat, flag, info)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatGetSize-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatGetSize","text":"MatGetSize(mat, m, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatGetValues-NTuple{6, Any}","page":"Home","title":"GridapPETSc.PETSC.MatGetValues","text":"MatGetValues(mat, m, idxm, n, idxn, v)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatMult-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatMult","text":"MatMult(mat, x, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatMultAdd-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.MatMultAdd","text":"MatMultAdd(mat, v1, v2, v3)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatMumpsSetCntl-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatMumpsSetCntl","text":"MatMumpsSetCntl(mat, icntl, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatMumpsSetIcntl-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatMumpsSetIcntl","text":"MatMumpsSetIcntl(mat, icntl, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatNullSpaceCreate-NTuple{5, Any}","page":"Home","title":"GridapPETSc.PETSC.MatNullSpaceCreate","text":"MatNullSpaceCreate(comm, has_cnst, n, vecs, sp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatNullSpaceCreateRigidBody-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatNullSpaceCreateRigidBody","text":"MatNullSpaceCreateRigidBody(coords, sp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatNullSpaceDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.MatNullSpaceDestroy","text":"MatNullSpaceDestroy(ns)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatScale-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatScale","text":"MatScale(mat, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatSetBlockSize-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatSetBlockSize","text":"MatSetBlockSize(mat, bs)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatSetNearNullSpace-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatSetNearNullSpace","text":"MatSetNearNullSpace(mat, nullsp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatSetValues-NTuple{7, Any}","page":"Home","title":"GridapPETSc.PETSC.MatSetValues","text":"MatSetValues(mat, m, idxm, n, idxn, v, addv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatView-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.MatView","text":"MatView(mat, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.MatZeroEntries-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.MatZeroEntries","text":"MatZeroEntries(mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PCFactorGetMatrix-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PCFactorGetMatrix","text":"PCFactorGetMatrix(ksp, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PCFactorSetMatSolverType-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PCFactorSetMatSolverType","text":"PCFactorSetMatSolverType(pc, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PCFactorSetUpMatSolverType-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PCFactorSetUpMatSolverType","text":"PCFactorSetUpMatSolverType(pc)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PCSetType-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PCSetType","text":"PCSetType(pc, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PCView-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PCView","text":"PCView(pc, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PETSC_VIEWER_DRAW_-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PETSC_VIEWER_DRAW_","text":"PETSC_VIEWER_DRAW_(comm)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PETSC_VIEWER_STDOUT_-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PETSC_VIEWER_STDOUT_","text":"PETSC_VIEWER_STDOUT_(comm)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscDataTypeFromString-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PetscDataTypeFromString","text":"PetscDataTypeFromString(name,ptype,found)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscDataTypeGetSize-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.PetscDataTypeGetSize","text":"PetscDataTypeGetSize(ptype,size)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscFinalize-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.PetscFinalize","text":"PetscFinalize()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscFinalized-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PetscFinalized","text":"PetscFinalized(flag)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscInitializeNoArguments-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.PetscInitializeNoArguments","text":"PetscInitializeNoArguments()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscInitializeNoPointers-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.PetscInitializeNoPointers","text":"PetscInitializeNoPointers(argc, args, filename, help)\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscInitialized-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PetscInitialized","text":"PetscInitialized(flag)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscObjectRegisterDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.PetscObjectRegisterDestroy","text":"PetscObjectRegisterDestroy(obj)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.PetscObjectRegisterDestroyAll-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.PetscObjectRegisterDestroyAll","text":"PetscObjectRegisterDestroyAll()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESCreate-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESCreate","text":"SNESCreate(comm, snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.SNESDestroy","text":"SNESDestroy(snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESGetKSP-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESGetKSP","text":"SNESGetKSP(snes, ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESSetFromOptions-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.SNESSetFromOptions","text":"SNESSetFromOptions(snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESSetFunction-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESSetFunction","text":"SNESSetFunction(snes, vec, fptr, ctx)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESSetJacobian-NTuple{5, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESSetJacobian","text":"SNESSetJacobian(snes, A, P, jacptr, ctx)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESSetType-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESSetType","text":"SNESSetType(snes, type)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESSolve-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESSolve","text":"SNESSolve(snes, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.SNESView-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.SNESView","text":"SNESView(snes, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecAXPBY-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.VecAXPBY","text":"VecAXPBY(y, alpha, beta, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecAXPY-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecAXPY","text":"VecAXPY(y, alpha, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecAYPX-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecAYPX","text":"VecAYPX(y, beta, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecAssemblyBegin-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.VecAssemblyBegin","text":"VecAssemblyBegin(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecAssemblyEnd-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.VecAssemblyEnd","text":"VecAssemblyEnd(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCopy-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCopy","text":"VecCopy(x, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCreateGhost-NTuple{6, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCreateGhost","text":"VecCreateGhost(comm, n, N, nghost, ghosts, vv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCreateGhostWithArray-NTuple{7, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCreateGhostWithArray","text":"VecCreateGhostWithArray(comm, n, N, nghost, ghosts, array, vv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCreateMPI-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCreateMPI","text":"VecCreateMPI(comm, n, N, v)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCreateSeq-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCreateSeq","text":"VecCreateSeq(comm, n, vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecCreateSeqWithArray-NTuple{5, Any}","page":"Home","title":"GridapPETSc.PETSC.VecCreateSeqWithArray","text":"VecCreateSeqWithArray(comm, bs, n, array, vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecDestroy-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.VecDestroy","text":"VecDestroy(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecDuplicate-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecDuplicate","text":"VecDuplicate(v, newv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetArray-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetArray","text":"VecGetArray(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetArrayRead-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetArrayRead","text":"VecGetArrayRead(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetArrayWrite-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetArrayWrite","text":"VecGetArrayWrite(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetLocalSize-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetLocalSize","text":"VecGetLocalSize(vec, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetSize-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetSize","text":"VecGetSize(vec, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGetValues-NTuple{4, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGetValues","text":"VecGetValues(x, ni, ix, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGhostGetLocalForm-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGhostGetLocalForm","text":"VecGhostGetLocalForm(g, l)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecGhostRestoreLocalForm-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecGhostRestoreLocalForm","text":"VecGhostRestoreLocalForm(g, l)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecNorm-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecNorm","text":"VecNorm(x, typ, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecPlaceArray-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecPlaceArray","text":"VecPlaceArray(vec, array)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecResetArray-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.VecResetArray","text":"VecResetArray(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecRestoreArray-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecRestoreArray","text":"VecRestoreArray(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecRestoreArrayRead-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecRestoreArrayRead","text":"VecRestoreArrayRead(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecRestoreArrayWrite-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecRestoreArrayWrite","text":"VecRestoreArrayWrite(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecScale-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecScale","text":"VecScale(x, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecSet-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecSet","text":"VecSet(x, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecSetOption-Tuple{Any, Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecSetOption","text":"VecSetOption(x, op, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecSetValues-NTuple{5, Any}","page":"Home","title":"GridapPETSc.PETSC.VecSetValues","text":"VecSetValues(x, ni, ix, y, iora)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.VecView-Tuple{Any, Any}","page":"Home","title":"GridapPETSc.PETSC.VecView","text":"VecView(vec, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"#GridapPETSc.PETSC.@PETSC_VIEWER_DRAW_SELF-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.@PETSC_VIEWER_DRAW_SELF","text":"@PETSC_VIEWER_DRAW_SELF\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"#GridapPETSc.PETSC.@PETSC_VIEWER_DRAW_WORLD-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.@PETSC_VIEWER_DRAW_WORLD","text":"@PETSC_VIEWER_DRAW_WORLD\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"#GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_SELF-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_SELF","text":"@PETSC_VIEWER_STDOUT_SELF\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"#GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD-Tuple{}","page":"Home","title":"GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_WORLD","text":"@PETSC_VIEWER_STDOUT_WORLD\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"#GridapPETSc.PETSC.@check_error_code-Tuple{Any}","page":"Home","title":"GridapPETSc.PETSC.@check_error_code","text":"@check_error_code expr\n\nCheck if expr returns an error code equal to zero(PetscErrorCode). If not, throw an instance of PetscError.\n\n\n\n\n\n","category":"macro"}]
}
