#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:49:14 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt

def F(noise_f_temp, t):
    """
    Computes the external driving force with added noise.

    Parameters:
        noise_f_temp (float): Noise factor for the force.
        t (float): Current time.

    Returns:
        float: The driving force at time t.
    """
    return F0 * np.cos(omega_f * t + noisiness_f * noise_f_temp)

def f(noise_f_temp, noise_temp, t, S):
    """
    Defines the system of differential equations for the stochastically driven oscillator.

    Parameters:
        noise_f_temp (float): Noise factor for the force.
        noise_temp (float): Noise factor for the damping.
        t (float): Current time.
        S (numpy.ndarray): State vector [position, velocity].

    Returns:
        numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp
    return dSdt

def euler(f, t0, tf, S0, h):
    """
    Implements the Euler method for solving ordinary differential equations.

    Parameters:
        f (function): Function defining the system of differential equations.
        t0 (float): Initial time.
        tf (float): Final time.
        S0 (numpy.ndarray): Initial state vector [position, velocity].
        h (float): Step size.

    Returns:
        numpy.ndarray: Array of time values.
        numpy.ndarray: Array of state vectors over time.
        numpy.ndarray: Array of noise values.
    """
    t_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)
        noise_f_temp = np.random.normal(loc=0, scale=1)
        noise_f_iso = np.append(noise_f_iso, noise_f_temp)
        
        x = x_values[-1, :]
        dxdt = f(noise_f_temp, noise_temp, t, x)
        
        x_new = x + h * dxdt

        x_values = np.concatenate((x_values, [x_new]), axis=0)
        t_values = np.append(t_values, t + h)
        t += h

    return t_values, x_values, noise_iso
# %%

# Define global parameters for the system
global r, omega, error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

m = 0.5  # Mass
k = 3.0  # Spring constant
gamma = 0.1  # Damping coefficient

r = gamma / (2 * m)  # Damping ratio
omega = (k / m) ** 0.5  # Natural frequency

x0 = 0.0  # Initial position
v0 = 3.0  # Initial velocity
t0 = 0.0  # Initial time
tf = 20.0  # Final time
h = 0.001  # Step size for Euler method
h_interpolate = 0.01  # Interpolation step size
S0 = np.array([x0, v0])  # Initial state vector
error_m = 1e-4  # Error tolerance (not used in Euler method)
F0 = 0  # Amplitude of driving force
noisiness = 43  # Noise level for the system
noisiness_f = 1  # Noise level for the driving force
omega_f = np.sqrt(k / m)  # Driving frequency
# %%

# Solve the system using Euler method
t_values, x_values, noise_iso = euler(f, t0, tf, S0, h)

# Plot the results
plt.plot(t_values, x_values[:, 0])
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'$\omega = \sqrt{\frac{k}{m}}$; Euler Method', usetex=False)
plt.grid(True)
plt.show()
