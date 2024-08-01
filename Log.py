#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:25:56 2024

@author: nathanngo
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

def F(t):
    """
    Computes the external driving force at a given time.
    
    Parameters:
    t (float): Current time.
    
    Returns:
    float: The computed force value.
    """
    return F0 * np.cos(omega_f * t)

def f(noise_temp, noisiness_temp, t, S):
    """
    Defines the system of differential equations for a damped harmonic oscillator.
    
    Parameters:
    noise_temp (float): Instantaneous random noise.
    noisiness_temp (float): Noise amplitude.
    t (float): Current time.
    S (array): State vector [position, velocity].
    
    Returns:
    array: Time derivative of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = - F(t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness_temp * noise_temp
    return dSdt

def RK45(noisiness_temp, f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta 45 method to solve a system of differential equations.
    
    Parameters:
    noisiness_temp (float): Noise amplitude.
    f (function): The system of differential equations to solve.
    t0 (float): Initial time.
    tf (float): Final time.
    S0 (array): Initial state vector.
    h (float): Initial step size.
    
    Returns:
    tuple: Arrays of time values, state vector values, and noise values.
    """
    t_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    n = 0
    noise_iso = np.empty(0, dtype=float)

    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)  # Generate random noise for each step
        noise_iso = np.append(noise_iso, noise_temp)
        n = n + 1
        x = x_values[-1, :]
        # Runge-Kutta 45 calculations
        k1 = h * f(noise_temp, noisiness_temp, t, x)
        k2 = h * f(noise_temp, noisiness_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_temp, noisiness_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_temp, noisiness_temp, t + (12 / 13) * h,
                   x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_temp, noisiness_temp, t + h,
                   x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_temp, noisiness_temp, t + (1 / 2) * h,
                   x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)

        while (error > error_m) or (error / error_m < 0.1):
            h = s * h  # Adjust step size based on the error
            k1 = h * f(noise_temp, noisiness_temp, t, x)
            k2 = h * f(noise_temp, noisiness_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_temp, noisiness_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_temp, noisiness_temp, t + (12 / 13) * h,
                       x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_temp, noisiness_temp, t + h,
                       x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_temp, noisiness_temp, t + (1 / 2) * h,
                       x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)

        # Append the new state and time values
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        t_values = np.append(t_values, t + h)
        t = t + h

    return t_values, x_values, noise_iso
# %%

global r, omega, error_m, omega_f, F0, h_interpolate, noisiness

m = 0.5  # Mass of the oscillator
k = 3.0  # Spring constant
gamma = 0.1  # Damping coefficient

r = gamma / (2 * m)  # Damping ratio
omega = (k / m) ** 0.5  # Natural frequency of the oscillator

x0 = 0.0  # Initial position
v0 = 3.0  # Initial velocity
t0 = 0.0  # Initial time
tf = 100.0  # Final time
h = 0.1  # Initial step size
T = 297  # Temperature (not used in the current code)
h_interpolate = 0.01  # Step size for interpolation
S0 = np.array([x0, v0])  # Initial state vector
error_m = 1e-5  # Error tolerance for RK45 method
F0 = 1  # Amplitude of the driving force
omega_f = np.sqrt(6)  # Driving frequency
# %%

noisiness = 40  # Initial noise amplitude
t_values_spline0 = np.arange(t0, tf, h_interpolate)
radius_spline0 = np.empty(len(t_values_spline0), dtype=float)

# Run 100 simulations with a given noise strength
for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline0)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline0)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline0 = np.vstack([radius_spline0, radius_temp])
    print(noisiness, i)
radius_spline0 = radius_spline0[1:]
radius_mean0 = np.mean(radius_spline0, axis=0)

# Repeat the process for different noise strengths
noisiness = 40.5
t_values_spline5 = np.arange(t0, tf, h_interpolate)
radius_spline5 = np.empty(len(t_values_spline5), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline5)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline5)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline5 = np.vstack([radius_spline5, radius_temp])
    print(noisiness, i)
radius_spline5 = radius_spline5[1:]
radius_mean5 = np.mean(radius_spline5, axis=0)

noisiness = 41
t_values_spline10 = np.arange(t0, tf, h_interpolate)
radius_spline10 = np.empty(len(t_values_spline10), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline10)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline10)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline10 = np.vstack([radius_spline10, radius_temp])
    print(noisiness, i)
radius_spline10 = radius_spline10[1:]
radius_mean10 = np.mean(radius_spline10, axis=0)

noisiness = 41.5
t_values_spline20 = np.arange(t0, tf, h_interpolate)
radius_spline20 = np.empty(len(t_values_spline20), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline20)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline20)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline20 = np.vstack([radius_spline20, radius_temp])
    print(noisiness, i)
radius_spline20 = radius_spline20[1:]
radius_mean20 = np.mean(radius_spline20, axis=0)

noisiness = 42
t_values_spline30 = np.arange(t0, tf, h_interpolate)
radius_spline30 = np.empty(len(t_values_spline30), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline30)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline30)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline30 = np.vstack([radius_spline30, radius_temp])
    print(noisiness, i)
radius_spline30 = radius_spline30[1:]
radius_mean30 = np.mean(radius_spline30, axis=0)

noisiness = 42.5
t_values_spline40 = np.arange(t0, tf, h_interpolate)
radius_spline40 = np.empty(len(t_values_spline40), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline40)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline40)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline40 = np.vstack([radius_spline40, radius_temp])
    print(noisiness, i)
radius_spline40 = radius_spline40[1:]
radius_mean40 = np.mean(radius_spline40, axis=0)

noisiness = 43
t_values_spline50 = np.arange(t0, tf, h_interpolate)
radius_spline50 = np.empty(len(t_values_spline50), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline50)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline50)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline50 = np.vstack([radius_spline50, radius_temp])
    print(noisiness, i)
radius_spline50 = radius_spline50[1:]
radius_mean50 = np.mean(radius_spline50, axis=0)

# Plotting the results for different noise strengths
plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('100 Runs of Noise Strength 43.0')
plt.grid(True)
plt.legend(loc='lower left')
plt.savefig('Figure.pdf')
plt.show()

# Plot the logarithm of the radius
plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title('100 Runs of Noise Strength 43.0')
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('Figure.pdf')
plt.show()

# %%
window_size = 200

# Calculate and plot rolling mean for different noise strengths
data = pd.DataFrame({'Time': t_values_spline0, 'ln<r>': np.log(radius_mean0)})
data['filtered0'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered0'], label='40')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline5, 'ln<r>': np.log(radius_mean5)})
data['filtered5'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered5'], label='40.5')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline10, 'ln<r>': np.log(radius_mean10)})
data['filtered10'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered10'], label='41')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline20, 'ln<r>': np.log(radius_mean20)})
data['filtered20'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered20'], label='41.5')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline30, 'ln<r>': np.log(radius_mean30)})
data['filtered30'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered30'], label='42')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline40, 'ln<r>': np.log(radius_mean40)})
data['filtered40'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered40'], label='42.5')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

data = pd.DataFrame({'Time': t_values_spline50, 'ln<r>': np.log(radius_mean50)})
data['filtered50'] = data['ln<r>'].rolling(window=window_size).mean()
plt.plot(data['Time'], data['filtered50'], label='43')
plt.xlabel('Noise Strength')
plt.ylabel('ln<r>')
plt.title('100 Runs of Different Noise Strengths (Smoothed)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

# %%
noisiness_op = 43
t_values_spline_op = np.arange(t0, tf, h_interpolate)
radius_spline_op = np.empty(len(t_values_spline_op), dtype=float)
t_values_op, x_values_op, noise_iso_op = RK45(noisiness_op, f, t0, tf, S0, h)

# Interpolate noise data for smoother plots
interpolator = interp1d(t_values_op[:-1], noise_iso_op, kind='cubic')
noise_iso_spline_op = interpolator(t_values_spline_op)

# Plot the autocorrelation of the noise
plot_acf(noise_iso_op)
# %%

noisiness_op = 43

t_values_spline_op = np.arange(t0, tf, h_interpolate)
radius_spline_op = np.empty(len(t_values_spline_op), dtype=float)
t_values_op, x_values_op, noise_iso_op = RK45(noisiness_op, f, t0, tf, S0, h)
noise_iso_op = noisiness_op * noise_iso_op

# Interpolate noise data for smoother plots
interpolator = interp1d(t_values_op[:-1], noise_iso_op, kind='cubic')
noise_iso_spline_op = interpolator(t_values_spline_op)

t_lag = np.arange(0, 300, 1)  # Time lag for autocorrelation
xi_product = np.empty(0, dtype=float)

# Compute autocorrelation of the noise
for lag in t_lag:
    xi_product_temp = np.empty(0, dtype=float)
    for i in range(len(noise_iso_spline_op)):
        if (i + lag < len(noise_iso_spline_op)):
            xi_product_temp = np.append(xi_product_temp, noise_iso_spline_op[i] * noise_iso_spline_op[i + lag])
    xi_product = np.append(xi_product, np.mean(xi_product_temp))

t_lag = h_interpolate * t_lag
plt.plot(t_lag, xi_product)
plt.xlabel(r'$|t - t\'|$', usetex=False)
plt.ylabel(r'$\langle \xi(t) \xi(t\') \rangle$', usetex=False)
plt.title('Noise Strength:43.0')
plt.grid(True)
plt.show()

print(xi_product)