module EventGeneration
include("injectors.jl")
include("detectors.jl")
using Reexport
@reexport using .Injectors
@reexport using .Detectors
end