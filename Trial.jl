using Random
using LinearAlgebra
using Interpolations
using FFTW
using Plots

# Define the necessary global variables
global error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f, cycle_count

function F(noise_f_temp, tau)
    return F0 * cos(omega_f * tau + noisiness_f * noise_f_temp)
end

function f(noise_f_temp, noise_temp, tau, S)
    dSdt = zeros(eltype(S), 2)  # Changed from zeros(eltype(S), length(S))
    dSdt[1] = S[2]
    dSdt[2] = F(noise_f_temp, tau) - 2 * zeta * S[2] - S[1] - noisiness * noise_temp
    return dSdt
end

function RK45(f, tau0, tauf, S0, h)
    tau_values = [tau0]
    x_values = [S0']
    tau = tau0
    n = 0
    noise_iso = Float64[]
    noise_f_iso = Float64[]

    global cycle_count
    cycle_count = 0
    noise_f_temp = 0.0

    while tau < tauf
        noise_temp = randn()
        push!(noise_iso, noise_temp)
        if cycle_count >= 3
            noise_f_temp += randn() * 2 * π
            cycle_count = 0
            push!(noise_f_iso, noisiness_f * noise_f_temp)
            println(tau, noisiness_f * noise_f_temp)
        end
        n += 1
        x = x_values[end, :]
        k1 = h * f(noise_f_temp, noise_temp, tau, x)
        k2 = h * f(noise_f_temp, noise_temp, tau + (1 / 4) * h, x .+ (1 / 4) .* k1)  # Element-wise operations
        k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8) * h, x .+ (3 / 32) .* k1 .+ (9 / 32) .* k2)  # Element-wise operations
        k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x .+ (1932 / 2197) .* k1 .- (7200 / 2197) .* k2 .+ (7296 / 2197) .* k3)  # Element-wise operations
        k5 = h * f(noise_f_temp, noise_temp, tau + h, x .+ (439 / 216) .* k1 .- 8 .* k2 .+ (3680 / 513) .* k3 .- (845 / 4104) .* k4)  # Element-wise operations
        k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x .- (8 / 27) .* k1 .+ 2 .* k2 .- (3544 / 2565) .* k3 .+ (1859 / 4104) .* k4 .- (11 / 40) .* k5)  # Element-wise operations
        x_new = x .+ (25 / 216) .* k1 .+ (1408 / 2565) .* k3 .+ (2197 / 4101) .* k4 .- (1 / 5) .* k5  # Element-wise operations
        z_new = x .+ (16 / 135) .* k1 .+ (6656 / 12825) .* k3 .+ (28561 / 56430) .* k4 .- (9 / 50) .* k5 .+ (2 / 55) .* k6  # Element-wise operations
        error = abs(z_new[1] - x_new[1])
        s = 0.84 * (error_m / error)^(1 / 4)
        # println(tau, h)

        while error > error_m
            h = s * h
            k1 = h * f(noise_f_temp, noise_temp, tau, x)
            k2 = h * f(noise_f_temp, noise_temp, tau + (1 / 4) * h, x .+ (1 / 4) .* k1)  # Element-wise operations
            k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8) * h, x .+ (3 / 32) .* k1 .+ (9 / 32) .* k2)  # Element-wise operations
            k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x .+ (1932 / 2197) .* k1 .- (7200 / 2197) .* k2 .+ (7296 / 2197) .* k3)  # Element-wise operations
            k5 = h * f(noise_f_temp, noise_temp, tau + h, x .+ (439 / 216) .* k1 .- 8 .* k2 .+ (3680 / 513) .* k3 .- (845 / 4104) .* k4)  # Element-wise operations
            k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x .- (8 / 27) .* k1 .+ 2 .* k2 .- (3544 / 2565) .* k3 .+ (1859 / 4104) .* k4 .- (11 / 40) .* k5)  # Element-wise operations
            x_new = x .+ (25 / 216) .* k1 .+ (1408 / 2565) .* k3 .+ (2197 / 4101) .* k4 .- (1 / 5) .* k5  # Element-wise operations
            z_new = x .+ (16 / 135) .* k1 .+ (6656 / 12825) .* k3 .+ (28561 / 56430) .* k4 .- (9 / 50) .* k5 .+ (2 / 55) .* k6  # Element-wise operations
            error = abs(z_new[1] - x_new[1])
            s = (error_m / error)^(1 / 5)
            # println(tau, h)
        end

        push!(x_values, x_new')  # Ensure correct dimensions
        push!(tau_values, tau + h)
        tau += h

        cycle_count += (omega_f * h) / (2 * π)
    end

    return tau_values, x_values, noise_iso
end

# %%

global error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f, cycle_count

x0 = 0.0
v0 = 0.0
tau0 = 0.0
tauf = 125.0
S0 = [x0, v0]

h = 0.1
h_interpolate = 0.01
error_m = 1e-5
zeta = 0.0

F0 = 1
noisiness = 0

omega_f = 1.0
cycle_count = 0
noisiness_f = 0

# Running the RK45 integration
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolating the results
t_values_spline = tau0:h_interpolate:tauf
x_values_spline = [LinearInterpolation(t_values_temp, x_values_temp[:, i])(t_values_spline) for i in 1:2]

# Fourier Transform of the results
X = fft(x_values_spline[1])
freqs = fftfreq(length(t_values_spline), h_interpolate)
X_noiseless = abs.(X[freqs .>= 0])
freqs_noiseless = freqs[freqs .>= 0]

plot(freqs_noiseless[1:30], X_noiseless[1:30], label="Noiseless")
xlabel("Frequency")
ylabel("Amplitude")
title("Fourier Transform; Underdamped; Noisy Force; Detuned to 0 Linewidth")
grid(true)
legend()
savefig("Relaxation_Decoherence_FFT.pdf")

noisiness_f = 10

# Running the RK45 integration with noise
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolating the results with noise
x_values_spline = [LinearInterpolation(t_values_temp, x_values_temp[:, i])(t_values_spline) for i in 1:2]

# Fourier Transform of the results with noise
X = fft(x_values_spline[1])
freqs = fftfreq(length(t_values_spline), h_interpolate)
X_noisy = abs.(X[freqs .>= 0])
freqs_noisy = freqs[freqs .>= 0]

plot(freqs_noiseless[10:30], X_noiseless[10:30], label="Noiseless")
plot!(freqs_noisy[10:30], X_noisy[10:30], label="Noisy")
xlabel("Frequency")
ylabel("Amplitude")
title("Fourier Transform; Undamped; Detuned to 0 Linewidth")
grid(true)
legend()
savefig("Relaxation_Decoherence_FFT.pdf")

omega_f = 1 + 0.015000000000000013
noisiness_f = 0

# Running the RK45 integration with new omega_f
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolating the results with new omega_f
x_values_spline = [LinearInterpolation(t_values_temp, x_values_temp[:, i])(t_values_spline) for i in 1:2]

# Fourier Transform of the results with new omega_f
X = fft(x_values_spline[1])
freqs = fftfreq(length(t_values_spline), h_interpolate)
X_noiseless = abs.(X[freqs .>= 0])
freqs_noiseless = freqs[freqs .>= 0]

noisiness_f = 10

# Running the RK45 integration with new omega_f and noise
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolating the results with new omega_f and noise
x_values_spline = [LinearInterpolation(t_values_temp, x_values_temp[:, i])(t_values_spline) for i in 1:2]

# Fourier Transform of the results with new omega_f and noise
X = fft(x_values_spline[1])
freqs = fftfreq(length(t_values_spline), h_interpolate)
X_noisy = abs.(X[freqs .>= 0])
freqs_noisy = freqs[freqs .>= 0]

plot(freqs_noiseless[10:30], X_noiseless[10:30], label="Noiseless")
plot!(freqs_noisy[10:30], X_noisy[10:30], label="Noisy")
xlabel("Frequency")
ylabel("Amplitude")
title("Fourier Transform; Undamped; Detuned to 2 Linewidth")
grid(true)
legend()
savefig("Relaxation_Decoherence_FFT.pdf")
