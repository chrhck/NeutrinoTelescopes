module PMTFrontEnd

include("spe_templates.jl")
include("pulse_templates.jl")
include("waveforms.jl")

using .PulseTemplates
using .Waveforms
using .SPETemplates


export PulseTemplates, Waveforms, SPETemplates


end