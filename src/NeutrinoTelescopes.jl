module NeutrinoTelescopes

using Reexport

include("types.jl")
include("utils.jl")

@reexport using .Types
@reexport using .Utils

include("photon_propagation/PhotonProp.jl")
include("pmt_frontend/PMTFrontEnd.jl")
include("event_generation/EventGeneration.jl")

include("surrogate_model/normalizing_flow.jl")

@reexport using .NormalizingFlow
@reexport using .PhotonPropagation
@reexport using .PMTFrontEnd

end