using NeutrinoTelescopes
using Plots
using GaussianProcesses
using Optim
using DataFrames
using Parquet

T = Float64
x = [T(300.0), T(365.0), T(400.0), T(450.0), T(585.0), T(800.0)]
y = [T(1), T(10.4), T(14.5), T(27.7), T(7.1), T(1)]

x = [T(365.0), T(400.0), T(450.0), T(585.0)]
y = [T(10.4), T(14.5), T(27.7), T(7.1)]


mZero = MeanZero()                 
#kern = Mat32Iso(3., 1.)
 
kern = RQ(5.,1.,1.)

logObsNoise = -4.0                        # log standard deviation of observation noise (this is optional)
gp = GP(x, log.(y),mZero,kern)   
plot(gp; xlabel="x", ylabel="y", title="Gaussian process", legend=false)  


optimize!(gp) 
plot(gp; xlabel="x", ylabel="y", title="Gaussian process", legend=false)  

wavelengths = 300.:5:800.

plot(wavelengths, exp.(predict_f(gp, wavelengths)[1]))
scatter!(x, y)

df = DataFrame(wavelength=wavelengths, abs_len=exp.(predict_f(gp, wavelengths)[1]))



write_parquet(joinpath(@__DIR__, "../assets/attenuation_length_straw_fit.parquet"), df)