using Gridap
using GridapDistributed
using PartitionedArrays
using SparseMatricesCSR, SparseArrays, LinearAlgebra

using Gridap.FESpaces

function subassemble_matrix(biform,assem,trial,test)
  u, v = get_trial_fe_basis(trial), get_fe_basis(test)
  data = collect_cell_matrix(trial,test,biform(u,v))

  rows = partition(get_rows(assem))
  cols = partition(get_cols(assem))
  mats = map(assemble_matrix, local_views(assem), data)

  A = PSparseMatrix(mats, rows, cols, nothing)
  return A
end

function subassemble_matrix_and_vector(biform,liform,assem,trial,test;assemble_vector=true)
  u, v = get_trial_fe_basis(trial), get_fe_basis(test)
  data = collect_cell_matrix_and_vector(trial,test,biform(u,v),liform(v),zero(trial))

  rows = partition(get_rows(assem))
  cols = partition(get_cols(assem))
  mats, vecs = map(assemble_matrix_and_vector, local_views(assem), data) |> tuple_of_arrays

  A = PSparseMatrix(mats, rows, cols, nothing)
  b = PVector(vecs, rows)

  if assemble_vector
    assemble!(b) |> wait
  end

  return A, b
end

function sa_mul!(C,A,B,α,β)
  rows, cols = axes(A)
  @assert PartitionedArrays.matching_local_indices(rows,axes(C,1))
  @assert PartitionedArrays.matching_local_indices(cols,axes(B,1))

  t = consistent!(B)

  if β != 1
    β != 0 ? rmul!(C, β) : fill!(C,zero(eltype(C)))
  end

  own_cols = own_to_local(cols)
  map(partition(C),partition(A),partition(B),own_cols) do C, A, B, cids
    mul!(C,view(A,:,cids),view(B,cids),α,1)
  end

  wait(t)

  ghost_cols = ghost_to_local(cols)
  map(partition(C),partition(A),partition(B),ghost_cols) do C, A, B, cids
    mul!(C,view(A,:,cids),view(B,cids),α,1)
  end

  assemble!(C) |> wait
  return C
end

#######################################################

u(x) = x[1] + x[2]

np = (2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4))

Ω = Triangulation(model)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V,u)

qdegree = 2
dΩ = Measure(Ω,qdegree)

f(x) = -Δ(u)(x)
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
l(v) = ∫(f⋅v)dΩ

# Assembler (CSR/CSC)
indexing = 1 # 0/1-based indexing
assem = SparseMatrixAssembler(SparseMatrixCSR{indexing,Float64,Int32},Vector{Float64},U,V)
#assem = SparseMatrixAssembler(SparseMatrixCSC{Float64,Int32},Vector{Float64},U,V)

# Assembled matrix and vector
A_as, b_as = assemble_matrix_and_vector(a,l,assem,U,V);

# Sub-Assembled matrix and vector
A_sa = subassemble_matrix(a,assem,U,V);
A_sa, b_sa = subassemble_matrix_and_vector(a,l,assem,U,V);

# Test sa multiplication
y_sa = pones(Float64,partition(axes(A_sa,2)))
x_sa = pzeros(Float64,partition(axes(A_sa,1)))
sa_mul!(x_sa,A_sa,y_sa,1.0,0.0)

y_as = pones(Float64,partition(axes(A_as,2)))
x_as = pzeros(Float64,partition(axes(A_as,1)))
mul!(x_as,A_as,y_as,1.0,0.0)

map(≈,own_values(x_sa), own_values(x_sa))
