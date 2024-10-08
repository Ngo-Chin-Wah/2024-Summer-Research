#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:53:31 2024

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

def RK45(f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta-Fehlberg (RK45) method for solving ordinary differential equations.

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
    n = 0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)
        noise_f_temp = np.random.normal(loc=0, scale=1)
        noise_f_iso = np.append(noise_f_iso, noise_f_temp)
        n += 1
        x = x_values[-1, :]
        k1 = h * f(noise_f_temp, noise_temp, t, x)
        k2 = h * f(noise_f_temp, noise_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)

        while error > error_m:
            h = s * h
            k1 = h * f(noise_f_temp, noise_temp, t, x)
            k2 = h * f(noise_f_temp, noise_temp, t + (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_f_temp, noise_temp, t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_f_temp, noise_temp, t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_f_temp, noise_temp, t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_f_temp, noise_temp, t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)

        x_values = np.concatenate((x_values, [x_new]), axis=0)
        t_values = np.append(t_values, t + h)
        t += h

    return t_values, x_values, noise_iso

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

# Global parameters for the system
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
h = 0.1  # Step size for RK45
h_interpolate = 0.01  # Interpolation step size
S0 = np.array([x0, v0])  # Initial state vector
error_m = 1e-6  # Error tolerance for RK45
F0 = 1  # Amplitude of driving force
noisiness = 0  # Noise level for the system
noisiness_f = 1  # Noise level for the driving force
omega_f = np.sqrt(k / m) + 1  # Driving frequency
# %%

# Solve the system using RK45 and Euler methods
t_values_RK45, x_values_RK45, noise_iso_RK45 = RK45(f, t0, tf, S0, h)
h = 0.0001  # Adjusted step size for Euler method
t_values_Euler, x_values_Euler, noise_iso_Euler = euler(f, t0, tf, S0, h)

# Plot the results for comparison
plt.plot(t_values_RK45, x_values_RK45[:, 0], label='RK45')
plt.plot(t_values_Euler, x_values_Euler[:, 0], label='Euler')
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'Stochastically Driven; $\omega>\sqrt{\frac{k}{m}}$', usetex=False)
plt.legend()
plt.grid(True)
plt.savefig('RK45 Euler Comparison.pdf')
plt.show()
