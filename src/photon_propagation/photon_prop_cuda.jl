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


export cuda_propagate_photons!, initialize_photon_arrays, process_output
export cherenkov_ang_dist, cherenkov_ang_dist_int
export propagate_source
export fit_photon_dist, make_photon_fits

using ..Utils
using ..Medium
using ..Spectral
using ..Emission
using ..Detection
using ..LightYield
using ..Types

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
    theta = uniform(T(-1), T(1))
    phi = uniform(T(0), T(2 * pi))
    sph_to_cart(theta, phi)
end


@inline initialize_wavelength(::T) where {T<:Spectrum} = throw(ArgumentError("Cannot initialize $T"))
@inline initialize_wavelength(spectrum::Monochromatic{T}) where {T} = spectrum.wavelength
@inline initialize_wavelength(spectrum::CuDeviceTexture{T, 1}) where {T} = @inbounds spectrum[rand(T)]

@inline function initialize_wavelength(spectrum::CherenkovSpectrum{T}) where {T}
    fast_linear_interp(rand(T), spectrum.knots, T(0), T(1))
end

@inline function initialize_photon_state(source::PointlikeIsotropicEmitter{T}, ::MediumProperties, spectrum_texture::CuDeviceTexture{T, 1}) where {T <: Real}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum_texture)
    pos = source.position
    dir = initialize_direction_isotropic(T)
    PhotonState(pos, dir, T(0), wl)
end

@inline function sample_cherenkov_direction(source::CherenkovEmitter{T}, medium::MediumProperties, wl::T) where {T <: Real}
    # Sample a direction of a "Cherenkov track". Coordinate system aligned with e_z
    track_dir::SVector{3, T} = sample_cherenkov_track_direction(T)
    
    # Rotate track coordinate system such, that e_z aligns with source direction
    track_dir = rotate_to_axis(source.direction, track_dir)

    # Sample a photon direction. Coordinate system aligned with e_z
    theta_cherenkov = get_cherenkov_angle(wl, medium)
    phi = uniform(T(0), T(2 * pi))
    
    ph_dir = sph_to_cart(theta_cherenkov, phi)

    # Rotate track coordinate system such, that e_z aligns with track direction
    ph_dir = rotate_to_axis(track_dir, ph_dir)
    return ph_dir
end


@inline function initialize_photon_state(source::ExtendedCherenkovEmitter{T, N}, medium::MediumProperties, spectrum_texture::CuDeviceTexture{T, 1}) where {T <:Real, N}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum_texture)

    long_pos = rand_gamma(T(source.long_param.a), T(1 / source.long_param.b), Float32) * source.long_param.lrad

    pos::SVector{3, T} = source.position .+ long_pos .* source.direction
    time = source.time + long_pos / T(c_vac_m_ns)
    
    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end

@inline function initialize_photon_state(source::PointlikeCherenkovEmitter{T, N}, medium::MediumProperties, spectrum_texture::CuDeviceTexture{T, 1}) where {T <:Real, N}
    #wl = initialize_wavelength(source.spectrum)
    wl = initialize_wavelength(spectrum_texture)

    pos::SVector{3, T} = source.position 
    time = source.time 
    
    ph_dir = sample_cherenkov_direction(source, medium, wl)

    PhotonState(pos, ph_dir, time, wl)
end


@inline function rotate_to_axis(old_dir::SVector{3,T}, new_dir::SVector{3,T}) where {T}
    #=
    New direction is relative to e_z. Axis of rotation defined by rotating e_z to old dir and applying
    that transformation to new_dir.

    Rodrigues rotation formula:
        ax = e_z x dir
        #theta = acos(dir * new_dir)
        theta = asin(|x|)

        axop = axis x new_dir
        rotated = new_dir * cos(theta) + sin(theta) * (axis x new_dir) + (1-cos(theta)) * (axis * new_dir) * axis
    =#

    if abs(old_dir[3]) == T(1)
        return @SVector[new_dir[1], copysign(new_dir[2], old_dir[3]), copysign(new_dir[3], old_dir[3])]
    end

    # Determine angle of rotation (cross product e_z and old_dir)
    # sin(theta) = | e_z x old_dir | = sqrt(1 - old_dir[3]^2)

    # sinthetasq = 1 - old_dir[3]*old_dir[3]
    sintheta = sqrt(fma(-old_dir[3], old_dir[3], 1))

    # Determine axis of rotation (cross product of e_z and old_dir )    
    ax1 = -old_dir[2] / sintheta
    ax2 = old_dir[1] / sintheta

    # rotated = operand .* cos(theta) + (cross(ax, operand) .* sin(theta)) +  (ax .* (1-cos(theta)) .* dot(ax, operand))

    # kappa = (ax1 * new_dir[1] + ax2 * new_dir[2]) * (1 - old_dir[3])
    kappa = fma(ax1, new_dir[1], ax2 * new_dir[2]) * (1 - old_dir[3])
    nd3sintheta = new_dir[3] * sintheta

    #new_x = new_dir[1] * old_dir[3] + ax2*nd3sintheta + ax1 * kappa
    #new_y = new_dir[2] * old_dir[3] - ax1*nd3sintheta + ax1 * kappa
    #new_z = new_dir[3] * old_dir[3] + (ax1*new_dir[2] - ax2*new_dir[1]) * sintheta

    new_x = fma(new_dir[1], old_dir[3], fma(ax2, nd3sintheta, ax1 * kappa))
    new_y = fma(new_dir[2], old_dir[3], fma(-ax1, nd3sintheta, ax2 * kappa))
    new_z = fma(new_dir[3], old_dir[3], sintheta * (fma(ax1, new_dir[2], -ax2 * new_dir[1])))

    # Can probably skip renormalizing
    norm = sqrt(new_x^2 + new_y^2 + new_z^2)

    return @SVector[new_x / norm, new_y / norm, new_z / norm]
end


@inline function update_direction(this_dir::SVector{3,T}) where {T}
    #=
    Update the photon direction using scattering function.
    =#

    # Calculate new direction (relative to e_z)
    cos_sca_theta = cuda_hg_scattering_func(T(0.99))
    sin_sca_theta = CUDA.sqrt(CUDA.fma(-cos_sca_theta, cos_sca_theta, 1))
    sca_phi = uniform(T(0), T(2 * pi))

    sin_sca_phi, cos_sca_phi = sincos(sca_phi)

    new_dir_1::T = cos_sca_phi * sin_sca_theta
    new_dir_2::T = sin_sca_phi * sin_sca_theta
    new_dir_3::T = cos_sca_theta

    rotate_to_axis(this_dir, @SVector[new_dir_1, new_dir_2, new_dir_3])

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
    source::U,
    spectrum_texture::CuDeviceTexture{T, 1},
    target_pos::SVector{3,T},
    target_r::T,
    ::Val{MediumProp}) where {T, U <: PhotonSource{T},MediumProp}

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

    target_rsq = target_r^2
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



function process_output(output::AbstractVector{T}, stack_pointers::AbstractVector{U}) where {T,U<:Integer,N}
    out_size = size(output, 1)
    stack_len = Int64(out_size / size(stack_pointers, 1))

    stack_starts = collect(1:stack_len:out_size)
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


function propagate_photons(
    source::PhotonSource,
    target::PhotonTarget,
    medium::MediumProperties,
    stack_len::Integer)

    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(1, 1, Float32)
    err_code = CuVector(zeros(Int32, 1))

    spectrum_vals = CuTextureArray(source.spectrum.knots)
    spectrum_texture = CuTexture(spectrum_vals; interpolation=CUDA.LinearInterpolation(), normalized_coordinates=true)

    kernel = @cuda launch=false cuda_propagate_photons!(
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, stack_len, Int64(0),
        source, spectrum_texture, target.position, target.radius, Val(medium))

    threads, blocks = CUDA.launch_configuration(kernel.fun, shmem=sizeof(Int64))
    
    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = initialize_photon_arrays(stack_len, blocks, Float32)

   
    kernel(
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim, err_code, stack_len, Int64(0),
        source, spectrum_texture, target.position, target.radius, Val(medium); threads=threads, blocks=blocks, shmem=sizeof(Int64))
    return (positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim)
end

function propagate_source(source::PhotonSource, distance, medium::MediumProperties, n_pmts=16, pmt_area=Float32((75e-3 / 2)^2*π))

    target_radius = 0.21f0
    distance = Float32(distance)
    target = DetectionSphere(@SVector[0.0f0, 0.0f0, distance], target_radius, n_pmts, pmt_area)

    avail_mem = CUDA.totalmem(collect(CUDA.devices())[1])
    max_total_stack_len = calculate_max_stack_size(0.5*avail_mem, blocks)

    @debug "Max stack size (90%): $max_total_stack_len"

    n_ph_gen = source.photons

    if n_ph_gen > max_total_stack_len
        @debug "Estimating acceptance fraction"
        test_nph = 1E5
        # Estimate required_stack_length
        stack_len = Int32(cld(1E5, blocks))

        test_source = PointlikeIsotropicEmitter(
            @SVector[0.0f0, 0.0f0, 0.0f0],
            0.0f0,
            Int64(test_nph),
            CherenkovSpectrum((300.0f0, 800.0f0), 20, medium),
        )
        positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = propagate_photons(
            test_source,
            target,
            medium,
            stack_len)

        n_ph_sim = Vector(n_ph_sim)[1]
        n_ph_det = sum(stack_idx .% stack_len)
        acc_frac = n_ph_det / n_ph_sim

        @debug "Acceptance fraction: $acc_frac"

        est_surv = acc_frac * n_ph_gen
        
        if est_surv > max_total_stack_len
            @warn "WARNING: Estimating more than $(max_total_stack_len) surviving photons, number of generated photons might be truncated" maxlog=3
            est_surv = max_total_stack_len
        end

        stack_len = max(Int32(cld(est_surv, blocks)), Int32(1E6))
    else
        stack_len = Int32(1E6)
    end

    positions, directions, wavelengths, dist_travelled, times, stack_idx, n_ph_sim = propagate_photons(
        source,
        target,
        medium,
        stack_len)


    n_ph_sim = Vector(n_ph_sim)[1]
    
    if all(stack_idx .== 0)
        @warn "No photons survived (distance=$distance, n_ph_gen: $n_ph_gen)"
        return DataFrame(), n_ph_sim
    end


    dist_travelled = process_output(Vector(dist_travelled), Vector(stack_idx))
    wavelengths = process_output(Vector(wavelengths), Vector(stack_idx))
    directions = process_output(Vector(directions), Vector(stack_idx))
    times = process_output(Vector(times), Vector(stack_idx))

    abs_weight = convert(Vector{Float64}, exp.(-dist_travelled ./ get_absorption_length.(wavelengths, Ref(medium))))

    ref_ix = get_refractive_index.(wavelengths, Ref(medium))
    c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
    # c_grp = get_group_velocity.(wavelengths, Ref(medium))


    #photon_times = dist_travelled ./ c_grp
    
    tgeo = (distance - target_radius) ./ (c_vac / get_refractive_index(800.0f0, medium))
    tres = (times .- tgeo .- source.time)

    (DataFrame(
        tres=tres,
        initial_directions=directions,
        ref_ix=ref_ix,
        abs_weight=abs_weight,
        dist_travelled=dist_travelled,
        wavelength=wavelengths),
    n_ph_sim)
end

#=
# Workaround for Flux breaking the RNG
function __init__()
    println("Doing initial prop")
    medium = make_cascadia_medium_properties(Float32)
    propagate_distance(10.0f0, medium, 100)
    nothing
end
=#

end # module