module NeutrinoTelescopes

using Reexport

include("types.jl")
include("utils.jl")

@reexport using .Types
@reexport using .Utils

include("photon_propagation/PhotonProp.jl")
include("pmt_frontend/PMTFrontEnd.jl")
include("event_generation/EventGeneration.jl")
include("surrogate_models/SurrogateModels.jl")


@reexport using .PhotonPropagation
@reexport using .PMTFrontEnd
@reexport using .SurrogateModels
@reexport using .EventGeneration
end
