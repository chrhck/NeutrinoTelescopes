module PhotonPropagationCuda
using StaticArrays
using BenchmarkTools
using LinearAlgebra
using CUDA
using Random
using SpecialFunctions
using DataFrames
using Unitful
using PhysicalConstants.CODATA2018
using StatsBase
using Logging
using PoissonRandom
using Rotations

export cuda_propagate_photons!, initialize_photon_arrays, process_output
export cuda_propagate_multi_target!
export cherenkov_ang_dist, cherenkov_ang_dist_int
export make_hits_from_photons, propagate_photons, run_photon_prop
export calc_time_residual

using ...Utils
using ...Types
using ..Medium
using ..Spectral
using ..Detection
using ..LightYield


const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)


"""
    uniform(minval::T, maxval::T) where {T}

Convenience function for sampling uniform values in [minval, maxval]
"""
@inline function uniform(minval::T, maxval::T) where {T}
    uni = rand(T)
    return minval + uni * (maxval - minval)
end

"""
    cuda_hg_scattering_func(g::Real)

CUDA-optimized version of Henyey-Greenstein scattering in one plane.

# Arguments
- `g::Real`: mean scattering angle

# Returns
- `typeof(g)` cosine of a scattering angle sampled from the distribution

"""
@inline function cuda_hg_scattering_func(g::Real)
    T = typeof(g)
    """Henyey-Greenstein scattering in one plane."""
    eta = rand(T)
    #costheta::T = (1 / (2 * g) * (1 + g^2 - ((1 - g^2) / (1 + g * (2 * eta - 1)))^2))
    costheta::T = (1 / (2 * g) * (CUDA.fma(g, g, 1) - (CUDA.fma(-g, g, 1) / (CUDA.fma(g, (CUDA.fma(2, eta, -1)), 1)))^2))
    CUDA.clamp(costheta, T(-1), T(1))
end


struct PhotonState{T}
    position::SVector{3, T}
    direction::SVector{3, T}
    time::T
    wavelength::T
end


"""
    initialize_direction_isotropic(T::Type)

Sample a direction isotropically

# Arguments
- `T::Type`: desired eltype of the return value

# Returns
- StaticVector{3, T}: Cartesian coordinates of the sampled direction

"""
@inline function initialize_direction_isotropic(T::Type)
    theta = acos(uniform(T(-1), T(1)))
    phi = uniform(T(0), T(2 * pi))
    sph_to_cart(theta, phi)
end


@inline function sample_cherenkov_direction(source::CherenkovEmitter{T}, medium::MediumProperties, wl::T) where {T <: Real}

    # Sample a photon direction. Assumes track is aligned with e_z
    theta_cherenkov = cherenkov_angle(wl, medium)
    phi = uniform(T(0), T(2 * pi))
    ph_dir = sph_to_cart(theta_cherenkov, phi)


    # Sample a direction of a "Cherenkov track". Assumes source direction is e_z
    track_dir::SVector{3, T} = sample_cherenkov_track_direction(T)

    # Rotate photon to track direction
    ph_dir = rot_from_ez_fast(track_dir, ph_dir)

    # Rotate track to source direction
    ph_dir = rot_from_ez_fast(source.direction, ph_dir)

    return ph_dir
end


@inline initialize_wavelength(::T) where {T<:Spectrum} = throw(ArgumentError("Cannot initialize $T"))
@inline initialize_wavelength(spectrum::Monochromatic{T}) where {T} = spectrum.wavelength
@inline initialize_wavelength(spectrum::CherenkovSpectrum{T, P}) where {T, P} = @inbounds spectrum.texture[rand(T)]


@inline function initialize_photon_state(source::PointlikeIsotropicEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T <: Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    dir = initialize_direction_isotropic(T)
    PhotonState(pos, dir, source.time, wl)
end

@inline function initialize_photon_state(source::PointlikeTimeRangeEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T <: Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    dir = initialize_direction_isotropic(T)
    time = uniform(source.time_range[1], source.time_range[2])
    PhotonState(pos, dir, time, wl)
end

@inline function initialize_photon_state(source::AxiconeEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T <: Real}
    wl = initialize_wavelength(spectrum)
    pos = source.position
    phi = uniform(T(0), T(2 * pi))
    theta = source.angle
    dir = rot_from_ez_fast(source.direction, sph_to_cart(theta, phi))

    PhotonState(pos, dir, source.time, wl)
end

@inline function initialize_photon_state(source::PencilEmitter{T}, ::MediumProperties, spectrum::Spectrum) where {T <: Real}
    wl = initialize_wavelength(spectrum)
    PhotonState(source.position, source.direction, source.time, wl)
end


@inline function initialize_photon_state(source::ExtendedCherenkovEmitter{T}, medium::MediumProperties, spectrum::Spectrum) where {T <:Real}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum)

    long_pos = rand_gamma(T(source.long_param.a), T(1 / source.long_param.b), Float32) * source.long_param.lrad

    pos::SVector{3, T} = source.position .+ long_pos .* source.direction
    time = source.time + long_pos / T(c_vac_m_ns)

    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end

@inline function initialize_photon_state(source::PointlikeCherenkovEmitter{T}, medium::MediumProperties, spectrum::Spectrum) where {T <:Real}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum)

    pos::SVector{3, T} = source.position
    time = source.time

    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end



@inline function update_direction(this_dir::SVector{3,T}, medium::MediumProperties) where {T}
    #=
    Update the photon direction using scattering function.
    =#

    # Calculate new direction (relative to e_z)
    cos_sca_theta = cuda_hg_scattering_func(mean_scattering_angle(medium))
    sin_sca_theta = CUDA.sqrt(CUDA.fma(-cos_sca_theta, cos_sca_theta, 1))
    sca_phi = uniform(T(0), T(2 * pi))

    sin_sca_phi, cos_sca_phi = sincos(sca_phi)

    new_dir_1::T = cos_sca_phi * sin_sca_theta
    new_dir_2::T = sin_sca_phi * sin_sca_theta
    new_dir_3::T = cos_sca_theta

    rot_from_ez_fast(this_dir, @SVector[new_dir_1, new_dir_2, new_dir_3])

end


@inline function update_position(this_pos, this_dir, this_dist_travelled, step_size)

    # update position
    for j in Int32(1):Int32(3)
        this_pos[j] = this_pos[j] + this_dir[j] * step_size
    end

    this_dist_travelled[1] += step_size
    return nothing
end

@inline function update_position(pos::SVector{3,T}, dir::SVector{3,T}, step_size::T) where {T}
    # update position
    #return @SVector[pos[j] + dir[j] * step_size for j in 1:3]
    return @SVector[CUDA.fma(step_size, dir[j], pos[j]) for j in 1:3]
end

@inline function check_intersection(pos::SVector{3,T}, dir::SVector{3,T}, target_pos::SVector{3,T}, target_rsq::T, step_size::T) where {T<:Real}

    dpos = pos .- target_pos

    a::T = dot(dir, dpos)
    pp_norm_sq::T = sum(dpos .^ 2)

    b = CUDA.fma(a, a, -pp_norm_sq + target_rsq)
    #b::Float32 = a^2 - (pp_norm_sq - target.radius^2)

    isec = b >= 0

    if !isec
        return false, NaN32
    end

    # Uncommon branch
    # Distance of of the intersection point along the line
    d = -a - sqrt(b)

    if (d > 0) & (d < step_size)
        return true, d
    else
        return false, NaN32
    end

end



function cuda_propagate_photons!(
    out_positions::CuDeviceVector{SVector{3,T}},
    out_directions::CuDeviceVector{SVector{3,T}},
    out_wavelengths::CuDeviceVector{T},
    out_dist_travelled::CuDeviceVector{T},
    out_times::CuDeviceVector{T},
    out_stack_pointers::CuDeviceVector{Int64},
    out_n_ph_simulated::CuDeviceVector{Int64},
    out_err_code::CuDeviceVector{Int32},
    stack_len::Int32,
    seed::Int64,
    source::PhotonSource,
    #spectrum_texture::CuDeviceTexture{T, 1},
    spectrum::Spectrum,
    target_pos::SVector{3,T},
    target_r::T,
    ::Val{MediumProp}) where {T, MediumProp}

    block = blockIdx().x
    thread = threadIdx().x
    blockdim = blockDim().x
    griddim = gridDim().x
    warpsize = CUDA.warpsize()
    # warp_ix = thread % warp
    global_thread_index::Int32 = (block - Int32(1)) * blockdim + thread

    cache = @cuDynamicSharedMem(Int64, 1)
    Random.seed!(seed + global_thread_index)


    this_n_photons::Int64 = cld(source.photons, (griddim * blockdim))

    medium::MediumProperties{T} = MediumProp

    target_rsq::T = target_r^2
    # stack_len is stack_len per block

    ix_offset::Int64 = (block - 1) * (stack_len) + 1
    @inbounds cache[1] = ix_offset

    safe_margin = max(0, (blockdim - warpsize))
    n_photons_simulated = Int64(0)


    @inbounds for i in 1:this_n_photons

        if cache[1] > (ix_offset + stack_len - safe_margin)
            CUDA.sync_threads()
            out_err_code[1] = -1
            break
        end

        photon_state = initialize_photon_state(source, medium, spectrum)

        dir::SVector{3,T} = photon_state.direction
        initial_dir = copy(dir)
        wavelength::T = photon_state.wavelength
        pos::SVector{3,T} = photon_state.position

        time = photon_state.time
        dist_travelled = T(0)

        sca_len::T = scattering_length(wavelength, medium)
        c_grp::T = group_velocity(wavelength, medium)


        steps::Int32 = 15
        for nstep in Int32(1):steps

            eta = rand(T)
            step_size::Float32 = -CUDA.log(eta) * sca_len

            # Check intersection with module

            isec, d = check_intersection(pos, dir, target_pos, target_rsq, step_size)

            if !isec
                pos = update_position(pos, dir, step_size)
                dist_travelled += step_size
                time += step_size / c_grp
                dir = update_direction(dir, medium)
                continue
            end

            # Intersected
            pos = update_position(pos, dir, d)
            dist_travelled += d
            time += d / c_grp

            stack_idx::Int64 = CUDA.atomic_add!(pointer(cache, 1), Int64(1))
            # @cuprintln("Thread: $thread, Block $block writing to $stack_idx")
            CUDA.@cuassert stack_idx <= ix_offset + stack_len "Stack overflow"

            out_positions[stack_idx] = pos
            out_directions[stack_idx] = initial_dir
            out_dist_travelled[stack_idx] = dist_travelled
            out_wavelengths[stack_idx] = wavelength
            out_times[stack_idx] = time
            CUDA.atomic_xchg!(pointer(out_stack_pointers, block), stack_idx)
            break
        end

        n_photons_simulated += 1

    end

    CUDA.atomic_add!(pointer(out_n_ph_simulated, 1), n_photons_simulated)
    out_err_code[1] = 0

    return nothing

end

function cuda_propagate_photons_no_local_cache!(
    out_positions::CuDeviceVector{SVector{3,T}},
    out_directions::CuDeviceVector{SVector{3,T}},
    out_wavelengths::CuDeviceVector{T},
    out_dist_travelled::CuDeviceVector{T},
    out_times::CuDeviceVector{T},
    out_stack_pointer::CuDeviceVector{Int64},
    out_n_ph_simulated::CuDeviceVector{Int64},
    out_err_code::CuDeviceVector{Int32},
    seed::Int64,
    source::PhotonSource,
    #spectrum_texture::CuDeviceTexture{T, 1},
    spectrum::Spectrum,
    target_pos::SVector{3,T},
    target_r::T,
    ::Val{MediumProp}) where {T, MediumProp}

    block = blockIdx().x
    thread = threadIdx().x
    blockdim = blockDim().x
    griddim = gridDim().x
    # warpsize = CUDA.warpsize()
    # warp_ix = thread % warp
    n_threads_total = (griddim * blockdim)
    global_thread_index::Int32 = (block - Int32(1)) * blockdim + thread

    Random.seed!(seed + global_thread_index)

    this_n_photons, remainder = divrem(source.photons, n_threads_total)

    if global_thread_index <= remainder
        this_n_photons +=1
    end

    medium::MediumProperties{T} = MediumProp

    target_rsq::T = target_r^2

    n_photons_simulated = Int64(0)

    @inbounds for i in 1:this_n_photons

        photon_state = initialize_photon_state(source, medium, spectrum)

        dir::SVector{3,T} = photon_state.direction
        initial_dir = copy(dir)
        wavelength::T = photon_state.wavelength
        pos::SVector{3,T} = photon_state.position

        time = photon_state.time
        dist_travelled = T(0)

        sca_len::T = scattering_length(wavelength, medium)
        c_grp::T = group_velocity(wavelength, medium)


        steps::Int32 = 15
        for nstep in Int32(1):steps

            eta = rand(T)
            step_size::Float32 = -CUDA.log(eta) * sca_len

            # Check intersection with module

            isec, d = check_intersection(pos, dir, target_pos, target_rsq, step_size)

            if !isec
                pos = update_position(pos, dir, step_size)
                dist_travelled += step_size
                time += step_size / c_grp
                dir = update_direction(dir, medium)
                continue
            end

            # Intersected
            pos = update_position(pos, dir, d)
            dist_travelled += d
            time += d / c_grp

            stack_idx::Int64 = CUDA.atomic_add!(pointer(out_stack_pointer, 1), Int64(1))
            CUDA.@cuassert stack_idx <= length(out_positions) "Stack overflow"

            out_positions[stack_idx] = pos
            out_directions[stack_idx] = initial_dir
            out_dist_travelled[stack_idx] = dist_travelled
            out_wavelengths[stack_idx] = wavelength
            out_times[stack_idx] = time
            break
        end

        n_photons_simulated += 1

    end

    CUDA.atomic_add!(pointer(out_n_ph_simulated, 1), n_photons_simulated)
    out_err_code[1] = 0

    return nothing

end




@inline function is_in(val, l, u)
    return (l < val) && (u > val)
end


#= IGNORE MODULE SHADOWING
function cuda_propagate_multi_target!(
    #= out_positions::CuDeviceVector{SVector{3,T}},
    out_directions::CuDeviceVector{SVector{3,T}},
    out_wavelengths::CuDeviceVector{T},
    out_dist_travelled::CuDeviceVector{T},
    out_times::CuDeviceVector{T},
    out_stack_pointers::CuDeviceVector{Int64},
    out_n_ph_simulated::CuDeviceVector{Int64},
    out_err_code::CuDeviceVector{Int32},
    stack_len::Int32,
     =#seed::Int64,
    # source::U,
    # spectrum_texture::CuDeviceTexture{T, 1},
    target_x::CuDeviceVector{T},
    target_y::CuDeviceVector{T},
    target_z::CuDeviceVector{T},
    targets_per_block::Int64,
    # target_r::T,
    ::Val{MediumProp}) where {T, MediumProp} #{T, U <: PhotonSource{T}, MediumProp}

    block = blockIdx().x
    thread = threadIdx().x
    blockdim = blockDim().x
    griddim = gridDim().x
    warpsize = CUDA.warpsize()
    # warp_ix = thread % warp
    global_thread_index::Int32 = (block - Int32(1)) * blockdim + thread

    cache = @cuDynamicSharedMem(Int64, 1)
    Random.seed!(seed + global_thread_index)

    n_targets = length(target_x)

    @cuassert targets_per_block * griddim == n_targets


    positions_x = @cuDynamicSharedMem(T, targets_per_block, sizeof(Int64))
    positions_y = @cuDynamicSharedMem(T, targets_per_block, sizeof(Int64) + sizeof(T)*targets_per_block)
    positions_z = @cuDynamicSharedMem(T, targets_per_block, sizeof(Int64) + 2*sizeof(T)*targets_per_block)

    @inbounds for i in thread:blockdim:targets_per_block
        positions_x[i] = target_x[(block-1)*targets_per_block+i]
    end

    @inbounds for i in thread:blockdim:targets_per_block
        positions_y[i] = target_y[(block-1)*targets_per_block+i]
    end

    @inbounds for i in thread:blockdim:targets_per_block
        positions_z[i] = target_z[(block-1)*targets_per_block+i]
    end

    this_n_photons::Int64 = cld(source.photons, blockdim)

    medium::MediumProperties{T} = MediumProp

    target_rsq = target_r^2

    ix_offset::Int64 = (block - 1) * (stack_len) + 1
    @inbounds cache[1] = ix_offset

    safe_margin = max(0, (blockdim - warpsize))
    n_photons_simulated = Int64(0)

    @inbounds for i in 1:this_n_photons

        if cache[1] > (ix_offset + stack_len - safe_margin)
            CUDA.sync_threads()
            out_err_code[1] = -1
            break
        end

        photon_state = initialize_photon_state(source, medium, spectrum_texture)

        dir::SVector{3,T} = photon_state.direction
        initial_dir = copy(dir)
        wavelength::T = photon_state.wavelength
        pos::SVector{3,T} = photon_state.position

        time = photon_state.time
        dist_travelled = T(0)

        sca_len::T = get_scattering_length(wavelength, medium)
        c_grp::T = get_group_velocity(wavelength, medium)


        steps::Int32 = 15
        for nstep in Int32(1):steps

            eta = rand(T)
            step_size::Float32 = -CUDA.log(eta) * sca_len


            endpos = update_position(pos, dir, step_size)

            for ntarget in 1:targets_per_block

                target_pos = SA[target_x[ntarget], target_y[ntarget], target_z[ntarget]]
                isec = false

                if sign(dir[1]) > 0 # positive x
                    isin_comp = is_in(target_pos[1], pos[1], endpos[1])
                else
                    isin_comp = is_in(target_pos[1], endpos[1], pos[1])

                if not isin_comp
                    continue
                end

                if sign(dir[2]) > 0 # positive x
                    isin_comp = is_in(target_pos[2], pos[2], endpos[2])
                else
                    isin_comp = is_in(target_pos[2], endpos[2], pos[2])

                if not isin_comp
                    continue
                end

                if sign(dir[3]) > 0 # positive x
                    isin_comp = is_in(target_pos[3], pos[3], endpos[3])
                else
                    isin_comp = is_in(target_pos[3], endpos[3], pos[3])

                if not isin_comp
                    continue
                end

                # Check intersection with module

                # a = dot(dir, (pos - target.position))
                # pp_norm_sq = norm(pos - target_pos)^2

                a::T = T(0)
                pp_norm_sq::T = T(0)

                for j in Int32(1):Int32(3)
                    dpos = (pos[j] - target_pos[j])
                    a += dir[j] * dpos
                    pp_norm_sq += dpos^2
                end


                b = CUDA.fma(a, a, -pp_norm_sq + target_rsq)
                #b::Float32 = a^2 - (pp_norm_sq - target.radius^2)

                isec = b >= 0

                if isec
                    # Uncommon branch
                    # Distance of of the intersection point along the line
                    d = -a - CUDA.sqrt(b)

                    isec = (d > 0) & (d < step_size)
                    if isec
                        # Step to intersection
                        pos = update_position(pos, dir, d)
                        #@cuprintln("Thread: $thread, Block $block, photon: $i, Intersected, stepped to $(pos[1])")
                        dist_travelled += d
                        time += d / c_grp
                    end
            else
                pos = update_position(pos, dir, step_size)
                dist_travelled += step_size
                time += step_size / c_grp
                dir = update_direction(dir)
            end

            #@cuprintln("Thread: $thread, Block $block, photon: $i, isec: $isec")
            if isec
                stack_idx::Int64 = CUDA.atomic_add!(pointer(cache, 1), Int64(1))
                # @cuprintln("Thread: $thread, Block $block writing to $stack_idx")
                CUDA.@cuassert stack_idx <= ix_offset + stack_len "Stack overflow"

                out_positions[stack_idx] = pos
                out_directions[stack_idx] = initial_dir
                out_dist_travelled[stack_idx] = dist_travelled
                out_wavelengths[stack_idx] = wavelength
                out_times[stack_idx] = time
                CUDA.atomic_xchg!(pointer(out_stack_pointers, block), stack_idx)
                break
            end
        end

        n_photons_simulated += 1

    end

    CUDA.atomic_add!(pointer(out_n_ph_simulated, 1), n_photons_simulated)
    out_err_code[1] = 0

    return nothing

end
=#



function process_output(output::AbstractVector{T}, stack_pointers::AbstractVector{U}) where {T,U<:Integer,N}

    output = Vector(output)

    out_size = length(output)
    stack_len = Int64(out_size / length(stack_pointers))

    stack_starts = 1:stack_len:out_size
    out_sum = sum(stack_pointers .% stack_len)

    coalesced = Vector{T}(undef, out_sum)
    ix = 1
    for i in eachindex(stack_pointers)
        sp = stack_pointers[i]
        if sp == 0
            continue
        end
        this_len = (sp - stack_starts[i]) + 1
        coalesced[ix:ix+this_len-1, :] = output[stack_starts[i]:sp]
        #println("$(stack_starts[i]:sp) to $(ix:ix+this_len-1)")
        ix += this_len

    end
    coalesced
end

function process_output(output::AbstractVector{T}, stack_pointer::Integer) where {T}
    return Vector(output[1:stack_pointer-1])
end



function initialize_photon_arrays(stack_length::Integer, blocks, type::Type)
    (
        CuVector(zeros(SVector{3,type}, stack_length * blocks)), # position
        CuVector(zeros(SVector{3,type}, stack_length * blocks)), # direction
        CuVector(zeros(type, stack_length * blocks)), # wavelength
        CuVector(zeros(type, stack_length * blocks)), # dist travelled
        CuVector(zeros(type, stack_length * blocks)), # time
        CuVector(zeros(Int64, blocks)), # stack_idx
        CuVector(zeros(Int64, 1)) # nphotons_simulated
    )
end


function initialize_photon_arrays(stack_length::Integer, type::Type)
    (
        CuVector(zeros(SVector{3,type}, stack_length)), # position
        CuVector(zeros(SVector{3,type}, stack_length)), # direction
        CuVector(zeros(type, stack_length)), # wavelength
        CuVector(zeros(type, stack_length)), # dist travelled
        CuVector(zeros(type, stack_length)), # time
        CuVector(ones(Int64, 1)), # stack_idx
        CuVector(zeros(Int64, 1)) # nphotons_simulated
    )
end


function calc_shmem(block_size)
    block_size * 7 * sizeof(Float32) #+ 3 * sizeof(Float32)
end


function calculate_gpu_memory_usage(stack_length, blocks)
    return sizeof(SVector{3, Float32}) * 2 * stack_length * blocks +
           sizeof(Float32) * 3 * stack_length * blocks +
           sizeof(Int32) * blocks +
           sizeof(Int64)
end

function calculate_max_stack_size(total_mem, blocks)
    return convert(Int32, floor((total_mem - sizeof(Int64) -  sizeof(Int32) * blocks) / (sizeof(SVector{3, Float32}) * 2 * blocks +  sizeof(Float32) * 3 * blocks)))
end

function calculate_max_stack_size(total_mem)
    one_event = sizeof(SVector{3, Float32}) * 2 + sizeof(Float32) * 3
    return Int64(fld(total_mem-2*(sizeof(Int64)), one_event))
end



function run_photon_prop(
    source::PhotonSource,
    target::PhotonTarget,
    medium::MediumProperties,
    spectrum::Spectrum
)

    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(1, 1, Float32)
    err_code = CuVector(zeros(Int32, 1))

    kernel = @cuda launch=false cuda_propagate_photons!(
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, Int32(1E6), Int64(0),
        source, spectrum, target.position, target.radius, Val(medium))

    blocks, threads = CUDA.launch_configuration(kernel.fun, shmem=sizeof(Int64))

    avail_mem = CUDA.totalmem(collect(CUDA.devices())[1])
    max_total_stack_len = calculate_max_stack_size(0.5*avail_mem, blocks)
    stack_len = Int32(cld(max_total_stack_len, blocks))

    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(stack_len, blocks, Float32)

    kernel(
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, stack_len, Int64(0),
        source, spectrum, target.position, target.radius, Val(medium); threads=threads, blocks=blocks, shmem=sizeof(Int64))

    return positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim
end

function run_photon_prop_no_local_cache(
    sources::AbstractVector{<:PhotonSource},
    target::PhotonTarget,
    medium::MediumProperties,
    spectrum::Spectrum
)
    avail_mem = CUDA.totalmem(collect(CUDA.devices())[1])
    max_total_stack_len = calculate_max_stack_size(0.5*avail_mem)
    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(max_total_stack_len, Float32)
    err_code = CuVector(zeros(Int32, 1))


    kernel = @cuda launch=false PhotonPropagationCuda.cuda_propagate_photons_no_local_cache!(
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, Int64(0),
        sources[1], spectrum, target.position, target.radius, Val(medium))

    blocks, threads = CUDA.launch_configuration(kernel.fun)

    # Can assign photons to sources by keeping track of stack_idx for each source
    for source in sources
        kernel(
            positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, Int64(0),
            source, spectrum, target.position, target.radius, Val(medium); threads=threads, blocks=blocks)
    end

    return positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim
end




function propagate_photons(
    sources::AbstractVector{<:PhotonSource},
    target::PhotonTarget,
    medium::MediumProperties,
    spectrum::Spectrum)

    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = run_photon_prop_no_local_cache(sources, target, medium, spectrum)

    stack_idx = Vector(stack_idx)[1]
    n_ph_sim = Vector(n_ph_sim)[1]

    if stack_idx == 0
        return DataFrame(), n_ph_sim
    end

    positions = process_output(positions, stack_idx)
    dist_travelled = process_output(dist_travelled, stack_idx)
    wavelengths = process_output(wavelengths, stack_idx)
    directions = process_output(directions, stack_idx)
    times = process_output(times, stack_idx)


    df = DataFrame(
        position=positions,
        direction=directions,
        wavelength=wavelengths,
        dist_travelled=dist_travelled,
        time=times)

    return df
end


function make_hits_from_photons(
    df::AbstractDataFrame,
    target::PhotonTarget,
    medium::MediumProperties,
    target_orientation::AbstractMatrix{<:Real})


    df[:, :pmt_id] = check_pmt_hit(df[:, :position], target, target_orientation)

    df = subset(df, :pmt_id => x -> x .> 0)

    df[!, :area_acc] = area_acceptance.(df[:, :position], Ref(target))

    abs_length = absorption_length.(df[:, :wavelength], Ref(medium))
    df[!, :abs_weight] = convert(Vector{Float64}, exp.(-df[:, :dist_travelled] ./ abs_length))

    df[!, :wl_acc] = p_one_pmt_acc.(df[:, :wavelength])

    df[!, :ref_ix] = refractive_index.(df[:, :wavelength], Ref(medium))

    df[!, :total_weight] = df[:, :area_acc] .* df[:, :wl_acc] .* df[:, :abs_weight]
    df
end

function calc_time_residual(
    df::AbstractDataFrame,
    source::PhotonSource,
    target::PhotonTarget,
    medium::MediumProperties,
)
    c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
    distance = norm(source.position .- target.position)
    tgeo = (distance - target.radius) ./ (c_vac / refractive_index(800.0f0, medium))
    df[!, :tres] = (df[:, :time] .- tgeo)
end


end # module
