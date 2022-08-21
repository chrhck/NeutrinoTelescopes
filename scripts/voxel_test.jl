using NeutrinoTelescopes
using Plots
using StaticArrays
using Base.Iterators
using CUDA
positions = make_detector_hex(6, 20, 100, 50, 2)

scatter([p[1] for p in positions], [p[2] for p in positions], [p[3] for p in positions])
unique([p[3] for p in positions])
scatter([p[1] for p in positions], [p[2] for p in positions])


boundary_x = extrema(p[1] for p in positions)
boundary_y = extrema(p[2] for p in positions)
boundary_z = extrema(p[3] for p in positions)

min_diff_x = minimum(diff(sort(unique(p[1] for p in positions))))
min_diff_y = minimum(diff(sort(unique(p[2] for p in positions))))
min_diff_z = minimum(diff(sort(unique(p[3] for p in positions))))

diff(sort(unique(p[2] for p in positions)))


steps_x = Int32(fld(boundary_x[2]-boundary_x[1], min_diff_x))
steps_y = Int32(fld(boundary_y[2]-boundary_y[1], min_diff_y))
steps_z = Int32(fld(boundary_z[2]-boundary_z[1], min_diff_z))

struct VoxelGrid{T<:Real} 
    n_steps::SVector{3, Int64}
    start::SVector{3, T}
    stop::SVector{3, T}
    Δ::SVector{3, T}
end

function VoxelGrid( n_steps::SVector{3, Int64}, start::SVector{3, T}, stop::SVector{3, T}) where {T<:Real}
    Δ = (stop .- start) ./ n_steps
    VoxelGrid(n_steps, start, stop, Δ)
end


function grid_index(position::SVector{3, <:Real}, grid::VoxelGrid) 
    offset = position .- grid.start
    ixs = Int64.(floor.(offset ./ grid.Δ))

    ixs
end

function grid_position(index::SVector{3, Int64}, grid::VoxelGrid)
    grid.start .+ index .* grid.Δ
end

struct DetectorGrid{T <: Real}
    positions::Array{SVector{3, T}, 3}
    is_filled::Array{Bool, 3}
end


function ray_box_intersect(position::SVector{3, T}, direction::SVector{3, T}, vmin::SVector{3, T}, vmax::SVector{3, T}) where {T <: Real}
    tmin = zero(T)
    tmax = zero(T)
    if (direction[1] >= 0) 
    	tmin = (vmin[1] - position[1]) / direction[1]
    	tmax = (vmax[1] - position[1]) / direction[1]
    else
    	tmin = (vmax[1] - position[1]) / direction[1]
    	tmax = (vmin[1] - position[1]) / direction[1]
    end
  
    if (direction[2] >= 0) 
        tymin = (vmin[2] - position[2]) / direction[2]
        tymax = (vmax[2] - position[2]) / direction[2]
    else
    	tymin = (vmax[2] - position[2]) / direction[2]
    	tymax = (vmin[2] - position[2]) / direction[2]
    end
    if ( (tmin > tymax) || (tymin > tmax) )
        return false, -1.
    end
       
    if (tymin > tmin)
        tmin = tymin
    end
    
	if (tymax < tmax)
        tmax = tymax
    end
    
	if (direction[3] >= 0)
       tzmin = (vmin[3] - position[3]) / direction[3]
       tzmax = (vmax[3] - position[3]) / direction[3]
    else
       tzmin = (vmax[3] - position[3]) / direction[3]
       tzmax = (vmin[3] - position[3]) / direction[3];
    end
    if ((tmin > tzmax) || (tzmin > tmax))
        return false, -1.
    end
    
    if (tzmin > tmin)
        tmin = tzmin;
    end
   
    if (tzmax < tmax)
        tmax = tzmax;
    end
    
    return true, tmin
end


function traverse_grid(position::SVector{3, T}, direction::SVector{3, T}, grid::VoxelGrid, det_grid::DetectorGrid{T}) where {T<:Real}

    l = @layout [a ; b ; c]
    p1 = plot(xlabel="x", ylabel="y", xlim=(-500, 500), ylim=(-500, 500))
    p2 = plot(xlabel="x", ylabel="z", xlim=(-500, 500), ylim=(-50, 1050))
    p3 = plot(xlabel="y", ylabel="z", xlim=(-500, 500), ylim=(-50, 1050))

    for ix in eachindex(det_grid.is_filled)
        if det_grid.is_filled[ix]
            dpos = det_grid.positions[ix]
            scatter!(p1, [dpos[1]], [dpos[2]], ms=5, label="", color=:red, alpha=0.2)
            scatter!(p2, [dpos[1]], [dpos[3]], ms=5, label="", color=:red, alpha=0.2)
            scatter!(p3, [dpos[2]], [dpos[3]], ms=5, label="", color=:red, alpha=0.2)
        end
    end

    isec, tmin = ray_box_intersect(position, direction, grid.start, grid.stop)

    if !isec
        return
    end

    if tmin <0
        tmin = 0
    end
    
    start = position .+ tmin .* direction

    gix = grid_index(start, grid)
    gpos = grid_position(gix, grid)
    step_signs = Int64.(sign.(direction))

    gpos_max = gpos .+ grid.Δ
    tmax = tmin .+ (gpos_max .- start) ./ direction

    tdelta = grid.Δ ./ abs.(direction)
    lin_ix = LinearIndices((1:grid.n_steps[1]+1, 1:grid.n_steps[2]+1, 1:grid.n_steps[3]+1))
    
    while(all(gix .>= SA[0, 0, 0]) && all(gix .<= grid.n_steps))

        isf = det_grid.is_filled[gix[1]+1, gix[2]+1, gix[3]+1]
        if isf
            color = :green
            dpos = det_grid.positions[gix[1]+1, gix[2]+1, gix[3]+1]
            scatter!(p1, [dpos[1]], [dpos[2]], ms=5, label="", color=:green, alpha=0.8)
            scatter!(p2, [dpos[1]], [dpos[3]], ms=5, label="", color=:green, alpha=0.8)
            scatter!(p3, [dpos[2]], [dpos[3]], ms=5, label="", color=:green, alpha=0.8)
        else
            color = :red
        end
        plot!(p1, rectangle(grid.Δ[1], grid.Δ[2], grid.start[1]+gix[1]*grid.Δ[1], grid.start[2]+gix[2]*grid.Δ[2]), opacity=0.5, label="", color=color)
        plot!(p2, rectangle(grid.Δ[1], grid.Δ[3], grid.start[1]+gix[1]*grid.Δ[1], grid.start[3]+gix[3]*grid.Δ[3]), opacity=0.5, label="", color=color)
        plot!(p3, rectangle(grid.Δ[2], grid.Δ[3], grid.start[2]+gix[2]*grid.Δ[2], grid.start[3]+gix[3]*grid.Δ[3]), opacity=0.5, label="", color=color)
       
        # Check intersecton in voxel

        @show lin_ix[gix[1]+1, gix[2]+1, gix[3]+1]
                
        if (tmax[1] < tmax[2])
            if (tmax[1] < tmax[3])
                gix = SA[gix[1] + step_signs[1], gix[2], gix[3]]
                tmax = SA[tmax[1] + tdelta[1], tmax[2], tmax[3]]
            else
                gix = SA[gix[1], gix[2], gix[3] + step_signs[3]]
                tmax = SA[tmax[1], tmax[2], tmax[3] + tdelta[3]]
            end
        else
            if (tmax[2] < tmax[3])
                gix = SA[gix[1], gix[2] + step_signs[2], gix[3]]
                tmax = SA[tmax[1], tmax[2] + tdelta[2], tmax[3]]       
            else
                gix = SA[gix[1], gix[2], gix[3] + step_signs[3]]
                tmax = SA[tmax[1], tmax[2], tmax[3] + tdelta[3]]
            end
        end
    end        

    plot!(p1, [position[1], position[1]+1500*direction[1]], [position[2], position[2]+1500*direction[2]], color=:black, lw=3)
    plot!(p2, [position[1], position[1]+1500*direction[1]], [position[3], position[3]+1500*direction[3]], color=:black, lw=3)
    plot!(p3, [position[2], position[2]+1500*direction[2]], [position[3], position[3]+1500*direction[3]], color=:black, lw=3)
    

    

    plot(p1, p2, p3, layout=l, size=(500, 1000))


end


grid = VoxelGrid((steps_x, steps_y, steps_z), SA[boundary_x[1], boundary_y[1], boundary_z[1]],  SA[boundary_x[2], boundary_y[2], boundary_z[2]])
grid

position = SA[-500., 0., 0.]
direction = sph_to_cart(deg2rad(20), deg2rad(50))

isec = ray_box_intersect(position, direction, grid.start, grid.stop)


rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

test_pos = SA[-220., -340., 50.]
gix = grid_index(test_pos, grid)

gpos = grid_position(gix, grid)

p = scatter([p[1] for p in positions], [p[2] for p in positions])
for (i, j) in product(0:grid.n_steps[1]-1, 0:grid.n_steps[2]-1)
    if i==gix[1] && j==gix[2]
        opac=7
    else
        opac=0.1
    end
    plot!(p, rectangle(grid.Δ[1], grid.Δ[2], grid.start[1]+i*grid.Δ[1], grid.start[2]+j*grid.Δ[2]), opacity=opac, label="", color=:red)
end

is_filled = zeros(Bool, grid.n_steps .+1)
voxel_ixs = grid_index.(positions, Ref(grid))
dpositions = Array{SVector{3, eltype(positions[1])}, 3}(undef, grid.n_steps .+1)

for pos in positions
    gix = grid_index(pos, grid)
    dpositions[gix[1]+1, gix[2]+1, gix[3]+1] = pos
end

for vxix in voxel_ixs
    is_filled[vxix[1]+1, vxix[2]+1, vxix[3]+1] = true
end

dgrid = DetectorGrid(dpositions, is_filled)

traverse_grid(position, direction, grid, dgrid)



is_filled