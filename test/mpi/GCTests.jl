include("../PLaplacianTests.jl")

function mem()
  Sys.free_memory() / 2^20
end

using Random
Random.seed!(1234)

iteration=0
function report_memory_and_random_gc(distribute,nparts)
  parts = distribute(LinearIndices((prod(nparts),)))
  map(parts) do part
    if (rand(1:length(parts)) == part)
      GC.gc()
      print("!!!GC.gc()ed on part $(part)!!!", "\n")
    end
    global iteration
    iteration = iteration + 1
    if (part==1)
      run(`ps -p $(getpid()) -o pid,comm,vsize,rss,size`)
    end
  end
end

function main_bis(distribute,nparts)
  main(distribute,nparts,:gmres,SubAssembledRows())
  report_memory_and_random_gc(parts)
end

options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 1.0 -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -pc_type jacobi -ksp_type gmres -snes_converged_reason"
GridapPETSc.Init(args=split(options))

nparts = (2,2)
NEXECS = 10
for i =1:NEXECS
  with_mpi() do distribute
    main(distribute,nparts)
  end
  if (i%2==0)
    GridapPETSc.gridap_petsc_gc()
  end
end

GridapPETSc.Finalize()
