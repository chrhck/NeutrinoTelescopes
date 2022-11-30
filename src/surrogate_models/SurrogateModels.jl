module SurrogateModels

using Reexport

include("rq_spline_flow.jl")
include("extended_cascade_model.jl")

@reexport using .RQSplineFlow
@reexport using .ExtendedCascadeModel
end
