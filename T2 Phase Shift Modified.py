#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:00:23 2024

@author: nathanngo
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq


def F(noise_f_temp, tau):
    """
    Computes the external force with noise.

    Parameters:
    noise_f_temp (float): Random noise component for the force.
    tau (float): Current time.
    
    Returns:
    float: The computed force value.
    """
    return F0 * np.cos(omega_f * tau + noisiness_f * noise_f_temp)


def f(noise_f_temp, noise_temp, tau, S):
    """
    Defines the system of differential equations for the oscillator.

    Parameters:
    noise_f_temp (float): Random noise component for the force.
    noise_temp (float): Instantaneous random noise.
    tau (float): Current time.
    S (array): State vector [position, velocity].

    Returns:
    array: Time derivative of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, tau) - 2 * zeta * S[1] - S[0] - noisiness * noise_temp
    return dSdt


def RK45(f, tau0, tauf, S0, h):
    """
    Implements the Runge-Kutta 45 method to solve a system of differential equations.
    
    Parameters:
    f (function): The system of differential equations to solve.
    tau0 (float): Initial time.
    tauf (float): Final time.
    S0 (array): Initial state vector.
    h (float): Initial step size.
    
    Returns:
    tuple: Arrays of time values, state vector values, and noise values.
    """
    tau_values = np.array([tau0])
    x_values = np.array([[S0[0], S0[1]]])
    tau = tau0
    n = 0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    global cycle_count
    cycle_count = 0
    noise_f_temp = 0

    while tau < tauf:
        noise_temp = np.random.normal(loc=0, scale=1)  # Generate random noise for each step
        noise_iso = np.append(noise_iso, noise_temp)
        if cycle_count >= 3:
            noise_f_temp += np.random.normal(loc=0, scale=1)  # New random phase shift
            cycle_count = 0  # Reset cycle count
            noise_f_iso = np.append(noise_f_iso, noisiness_f * noise_f_temp)
            print(tau, noisiness_f * noise_f_temp)
        n += 1
        x = x_values[-1, :]

        # Runge-Kutta 45 calculations
        k1 = h * f(noise_f_temp, noise_temp, tau, x)
        k2 = h * f(noise_f_temp, noise_temp, tau +
                   (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8)
                   * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x +
                   (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, tau + h, x + (439 / 216)
                   * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x - (8 / 27) *
                   k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * \
                k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + \
                (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print(tau, h)

        while (error > error_m):
            h = s * h
            k1 = h * f(noise_f_temp, noise_temp, tau, x)
            k2 = h * f(noise_f_temp, noise_temp, tau +
                       (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8)
                       * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x +
                       (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_f_temp, noise_temp, tau + h, x + (439 / 216)
                       * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x - (8 / 27) *
                       k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * \
                    k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + \
                    (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)
            print(tau, h)

        # Append the new state and time values
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, tau + h)
        tau = tau + h

        cycle_count += (omega_f * h) / (2 * np.pi)  # Update the cycle count

    return tau_values, x_values, noise_iso


# %%

global error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f, phase_shift, cycle_count

x0 = 0.0  # Initial position
v0 = 10.0  # Initial velocity
tau0 = 0.0  # Initial time
tauf = 100.0  # Final time
S0 = np.array([x0, v0])  # Initial state vector
h = 0.01  # Initial step size
h_interpolate = 0.01  # Step size for interpolation
error_m = 1e-6  # Error tolerance for RK45 method
zeta = 0.5  # Damping factor
F0 = 1.0  # Amplitude of the driving force
noisiness = 0  # Noise level for the system
omega_f = 5.0  # Driving frequency
cycle_count = 0  # Initialize cycle count
# %%
noisiness_f = 0  # Noise level for the force

# Run the RK45 solver without noise in the force
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolation for smooth plotting
t_values_spline = np.arange(tau0, tauf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline = interpolator(t_values_spline)

# Perform Fourier Transform on the interpolated position values
freqs_noiseless = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline)) * h_interpolate
X_noiseless = np.abs(X)

noisiness_f = 0.5  # Increase noise level for the force

# Run the RK45 solver with noise in the force
t_values_temp, x_values_temp, noise_iso = RK45(f, tau0, tauf, S0, h)

# Interpolation for smooth plotting
t_values_spline = np.arange(tau0, tauf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline = interpolator(t_values_spline)

# Perform Fourier Transform on the interpolated position values with noise
freqs_noisy = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline)) * h_interpolate
X_noisy = np.abs(X)

# %%
# Plot the Fourier Transform result with and without noise for comparison
# plt.plot(freqs_noisy[14700:-14700], X_noisy[14700:-14700], label='Noisy')
plt.plot(freqs_noiseless[4900:-4900], X_noiseless[4900:-4900], label='Noiseless')
plt.xlabel(r'Frequency', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(r'Fourier Transform; Critically Damped; Detuned to 1 Linewidth', usetex=True)
plt.grid(True)
plt.legend()
plt.savefig('Figure.pdf')
plt.show()

plt.plot(t_values_spline, x_values_spline)
plt.show()

# %%
freqs_new = freqs_noiseless[4950:-4950]
abs_new = np.abs(X_noiseless)[4950:-4950]
freqs_spline = np.arange(np.min(freqs_new), np.max(freqs_new), 0.001)

# Interpolate and plot the smoothed Fourier Transform result
interpolator = interp1d(freqs_new, abs_new, kind='cubic')
abs_spline = interpolator(freqs_spline)

plt.plot(freqs_spline, abs_spline)
plt.show()

# Determine the range of frequencies where the amplitude is above half the maximum value
abs_max = np.max(abs_spline)
freqs_above = np.empty(0, dtype=float)

for i in range(len(freqs_spline)):
    if (abs_spline[i] >= 0.5 * abs_max):
        print(freqs_spline[i])
        freqs_above = np.append(freqs_above, freqs_spline[i])

linewidth = freqs_above[-1] - freqs_above[0]
print('Linewidth:', linewidth)
