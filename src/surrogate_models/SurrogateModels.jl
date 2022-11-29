module SurrogateModels

using Reexport

include("normalizing_flow.jl")

@reexport using .NormalizingFlow
end