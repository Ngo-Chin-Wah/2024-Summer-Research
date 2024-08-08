#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:59:10 2024

@author: nathanngo
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def F(noise_f_temp, t):
    """
    Computes the external force with noise.

    Parameters:
    noise_f_temp (float): Random noise component for the force.
    tau (float): Current time.
    
    Returns:
    float: The computed force value.
    """
    return F0 * np.cos(omega_f * t + noisiness_f * noise_f_temp)

def f(noise_f_temp, noise_temp, t, S):
    """
    Defines the system of differential equations for the oscillator.

    Parameters:
        noise_f_temp (float): Noise affecting the driving force.
        noise_temp (float): Noise affecting the system's state.
        t (float): Time variable.
        S (numpy.ndarray): State vector [position, velocity].

    Returns:
        numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp
    return dSdt

def RK45(f, t0, tf, S0, h):
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
    tau_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    n = 0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    global cycle_count
    cycle_count = 0
    noise_f_temp = 0

    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)  # Generate random noise for each step
        noise_iso = np.append(noise_iso, noise_temp)
        if cycle_count >= 3:
            noise_f_temp += np.random.normal(loc=0, scale=1)  # New random phase shift
            cycle_count = 0  # Reset cycle count
            noise_f_iso = np.append(noise_f_iso, noisiness_f * noise_f_temp)
            print(t, noisiness_f * noise_f_temp)
        n += 1
        x = x_values[-1, :]

        # Runge-Kutta 45 calculations
        k1 = h * f(noise_f_temp, noise_temp, t, x)
        k2 = h * f(noise_f_temp, noise_temp, t +
                   (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8)
                   * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h, x +
                   (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, t + h, x + (439 / 216)
                   * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h, x - (8 / 27) *
                   k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * \
                k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + \
                (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print(t, h)

        while (error > error_m):
            h = s * h
            k1 = h * f(noise_f_temp, noise_temp, t, x)
            k2 = h * f(noise_f_temp, noise_temp, t +
                       (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8)
                       * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h, x +
                       (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_f_temp, noise_temp, t + h, x + (439 / 216)
                       * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h, x - (8 / 27) *
                       k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * \
                    k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + \
                    (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)
            print(t, h)

        # Append the new state and time values
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, t + h)
        t += h

        cycle_count += (omega_f * h) / (2 * np.pi)  # Update the cycle count

    return tau_values, x_values, noise_iso
# %%

global r, omega, error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

# System parameters
m = 0.1
k = 60.0
gamma = 0.5

r = gamma / (2 * m)
omega = (k / m) ** 0.5

# Initial conditions and other parameters
x0 = 0.0
v0 = 3.0
t0 = 0.0
tf = 100.0
h = 0.1
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-6
F0 = 1
noisiness = 0
noisiness_f = 0
linewidth = 1.4079999999992197
omega_f = omega + 2 * np.pi * 0.0 * linewidth

# Run the RK45 solver without noise in the force
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolation for smooth plotting
t_values_spline = np.arange(t0, tf, h_interpolate)
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
t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)

# Interpolation for smooth plotting
t_values_spline = np.arange(t0, tf, h_interpolate)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)
interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
v_values_spline = interpolator(t_values_spline)

# Perform Fourier Transform on the interpolated position values with noise
freqs_noisy = np.fft.fftshift(np.fft.fftfreq(len(t_values_spline), d=h_interpolate))
X = np.fft.fftshift(np.fft.fft(x_values_spline)) * h_interpolate
X_noisy = np.abs(X)
# %%

plt.plot(freqs_noisy[4500:-4500], X_noisy[4500:-4500], label='Noisy')
plt.plot(freqs_noiseless[4500:-4500], X_noiseless[4500:-4500], label='Noiseless')
plt.xlabel(r'Frequency', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(r'Fourier Transform; Underdamped; Detuned to 0 Linewidth', usetex=True)
plt.grid(True)
plt.legend()
plt.savefig('Figure.pdf')
plt.show()

plt.plot(t_values_spline, x_values_spline)
plt.show()

# %%
freqs_new = freqs_noiseless[4000:-5000]
abs_new = np.abs(X_noiseless)[4000:-5000]
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
