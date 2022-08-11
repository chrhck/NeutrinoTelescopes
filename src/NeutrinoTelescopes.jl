module NeutrinoTelescopes

include("types.jl")
include("utils.jl")
include("photon_propagation/medium.jl")
include("photon_propagation/spectrum.jl")
include("photon_propagation/lightyield.jl")
include("photon_propagation/emission.jl")
include("photon_propagation/detection.jl")
include("photon_propagation/photon_prop_cuda.jl")
include("photon_propagation/modelling.jl")

include("pmt_frontend/PMTFrontEnd.jl")

include("event_generation/EventGeneration.jl")

using .Types
using .Utils
using .Medium
using .Spectral
using .Emission
using .LightYield


export Medium, Spectral, Emission, LightYield, Modelling, Utils, Detection, Types, PhotonPropagationCuda

end