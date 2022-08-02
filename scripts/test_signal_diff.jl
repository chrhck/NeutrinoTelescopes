using DSP
using Enzyme
using Plots
	
function gen_signal(x::Number, freq1::Number, freq2::Number)
    T = promote_type(typeof(x), typeof(freq1), typeof(freq2))
    @show T
    T(sin(2*π*x*freq1) + sin(2*π*x*freq2))
end


function filt_signal(freq1, freq2)
    dt = 0.01
    timesteps = 0.:dt:5.
    signal = gen_signal.(timesteps, freq1, freq2)

    filter = digitalfilter(Lowpass(1., fs=1/dt), Butterworth(5))
    filt_sig = filt(filter, signal)


    timesteps, signal, filt_sig
    
end

timesteps, signal, filt_sig = filt_signal(1, 2)

plot(timesteps, signal)
plot!(timesteps, filt_sig)

autodiff(filt_signal, Active(1.), Active(2.))
	
filter = digitalfilter(Lowpass(1, fs=1/0.01), Butterworth(5))

so = convert(SecondOrderSections, filter)
