# PETScFieldSplit 
# MF (multifield) is a vector where each element is a SingleFieldFESpace or DistributedSingleFieldFESpace, eg: [U,P]
# tags is a vector containing the names of the field (in order), eg: ["vel", "pres"]
# show_idx is Bool vector, if true it prints the indexes
struct PETScFieldSplit
    MF
    tags::Vector{String}
    show_idx::Bool
end

# PETScFieldSplit constructor from Multifield
function PETScFieldSplit(MF::Union{MultiFieldFESpace, GridapDistributed.DistributedMultiFieldFESpace}, tags::Vector{String}; show_idx = false)
  n_fields = length(MF)
  n_tags = length(tags)
  @assert n_tags == n_fields #Verify that at each field a name is assigned
  M = [MF[1]] #Vector of SingleFieldFESpace
  for i = 2:1:n_fields
     M = [M...,MF[i]]
  end
    PETScFieldSplit(M, tags, show_idx)
end

function PETScFieldSplit(MF::Union{Vector{SingleFieldFESpace}, Vector{<:GridapDistributed.DistributedSingleFieldFESpace}}, tags; show_idx = false)
  n_fields = length(MF)
  n_tags = length(tags)
  @assert n_tags == n_fields
  PETScFieldSplit(MF, tags, show_idx)
end




function field_dof_split(U)
  X = MultiFieldFESpace(U)
  field_dof_split(X,U)
end

# It allows to obtain the dof of each field over each processor - dont know if the best/most elegant way
function field_dof_split(X, U)
  @assert length(X) == length(U)
    xrows = X.gids.partition
    urows = Any[]
    for Ui in X
      urowsi = Ui.gids.partition
      push!(urows, urowsi)
    end

    offset = 0
    ulrows = Any[]
    for urowsi in urows
      ulrowsi = map_parts(urowsi,xrows) do urowsij,xrows
      num_dof_Ui = length(urowsij.oid_to_lid)
      uloc_idx = collect(1+offset:1:num_dof_Ui + offset)
        uri = xrows.lid_to_gid[xrows.oid_to_lid][uloc_idx]
        offset = num_dof_Ui+offset
        return uri
    end
    #It works only in MPI, non-MPI do not have .part fields
    # .-1 to match the numbering in PETSc that starts from 0
    ur = ulrowsi.part .-1
    push!(ulrows,ur)
  end
    return ulrows
end



function set_fieldsplit(sol, SplitField::PETScFieldSplit)

  #Get PC-string name
  pc = Ref{GridapPETSc.PETSC.PC}()
  pctype = Ref{Ptr{Cstring}}()
  
  #Get the PC, if the problem is linear ksp or non linear snes
  if typeof(sol) <:Base.RefValue{GridapPETSc.PETSC.KSP}
    ksp = sol
  elseif typeof(sol) <:Base.RefValue{GridapPETSc.PETSC.SNES}
    snes = sol
    ksp      = Ref{GridapPETSc.PETSC.KSP}()
    @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
  end

  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
  @check_error_code GridapPETSc.PETSC.PCGetType(pc[], pctype)

  pc_ptr_conv = reinterpret(Ptr{UInt8}, pctype[])
  GC.@preserve pc_name = unsafe_string(pc_ptr_conv)
    
    @assert  pc_name == "fieldsplit" #Check that the preconditioner requires splitting fields
    n_tags = length(SplitField.tags)
    U_Parray = field_dof_split(SplitField.MF)
    for i = 1:1:n_tags
      ISU =   PETScIS(U_Parray[i])
      @check_error_code GridapPETSc.PETSC.PCFieldSplitSetIS(pc[], SplitField.tags[i], ISU.is[])
        if SplitField.show_idx
          @check_error_code GridapPETSc.PETSC.ISView(ISU.is[], GridapPETSc.PETSC.@PETSC_VIEWER_STDOUT_SELF)
        end
    end
  
    
  end