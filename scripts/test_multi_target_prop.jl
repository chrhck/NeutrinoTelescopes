using NeutrinoTelescopes
using CUDA
using Cthulhu

medium = make_cascadia_medium_properties(Float32)

ntargets = 10

tx = CuVector(zeros(Float32, ntargets))
ty = CuVector(zeros(Float32, ntargets))
tz = CuVector(zeros(Float32, ntargets))


threads = 12
blocks = 5

targets_per_block = Int64(cld(ntargets, blocks))

shmem = sizeof(Int64) + 3*sizeof(Float32)*targets_per_block

@cuda threads=threads blocks=blocks shmem=shmem cuda_propagate_multi_target!(1, tx, ty, tz, targets_per_block, Val(medium))
