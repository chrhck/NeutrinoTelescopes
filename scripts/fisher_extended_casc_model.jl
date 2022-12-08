using NeutrinoTelescopes
using Flux
using CUDA

using BSON: @save, @load


model_path = joinpath(@__DIR__, "../assets/rq_spline_model.bson")
@load model_path model hparams opt tf_dict


p = Particle(pos, dir, 0.0, 1E5, PEPlus)
target = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32),
    UInt16(1)
)


input = calc_flow_inputs([p], [target], traf)
