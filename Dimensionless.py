#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 11 15:34:47 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt


def F(noise_f_temp, tau):
    """
    Computes the external force with noise.

    Args:
        noise_f_temp (float): Random noise component for the force.
        tau (float): Current time.

    Returns:
        float: The computed force value.
    """
    return F0 * np.cos(omega_f * tau + noisiness_f * noise_f_temp)


def f(noise_f_temp, noise_temp, tau, S):
    """
    Defines the system of differential equations for the oscillator.

    Args:
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

    Args:
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

    while tau < tauf:
        noise_temp = np.random.normal(loc=0, scale=1)  # Generate random noise for each step
        noise_iso = np.append(noise_iso, noise_temp)
        noise_f_temp = np.random.normal(loc=0, scale=1)  # Generate random phase shift noise
        noise_f_iso = np.append(noise_f_iso, noise_f_temp)
        n = n + 1
        x = x_values[-1, :]
        # Runge-Kutta 45 calculations
        k1 = h * f(noise_f_temp, noise_temp, tau, x)
        k2 = h * f(noise_f_temp, noise_temp, tau + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(noise_f_temp, noise_temp, tau + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print(tau, h)

        while (error > error_m):
            h = s * h  # Adjust step size based on the error
            k1 = h * f(noise_f_temp, noise_temp, tau, x)
            k2 = h * f(noise_f_temp, noise_temp, tau + (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(noise_f_temp, noise_temp, tau + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(noise_f_temp, noise_temp, tau + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(noise_f_temp, noise_temp, tau + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(noise_f_temp, noise_temp, tau + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = (error_m / error) ** (1 / 5)
            print(tau, h)

        # Append the new state and time values
        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, tau + h)
        tau = tau + h
    return tau_values, x_values, noise_iso
# %%

global error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

x0 = 0.0  # Initial position
v0 = 3.0  # Initial velocity
tau0 = 0.0  # Initial time
tauf = 70.0  # Final time
S0 = np.array([x0, v0])  # Initial state vector

h = 0.1  # Initial step size
h_interpolate = 0.01  # Step size for interpolation
error_m = 1e-6  # Error tolerance for RK45 method

F0 = 1  # Amplitude of the driving force
omega_f = 10  # Driving frequency

noisiness = 0  # Noise level for the system
noisiness_f = 0  # Noise level for the force

# %%
zeta = 0.1  # Damping factor for the underdamped case
# Run the RK45 solver for the underdamped and driven cases
t_values, x_values, noise_iso = RK45(f, tau0, tauf, S0, h)
t_values_highfreq, x_values_highfreq, noise_iso_highfreq = RK45(f, tau0, tauf, S0, h)

# Plot the results of the undriven and driven cases
plt.plot(t_values, x_values[:, 0], label='Undriven')
plt.plot(t_values_highfreq, x_values_highfreq[:, 0], label='Driven by High Frequency Force')
plt.xlabel(r'$\tau(\frac{1}{\omega_m})$', usetex=True)
plt.ylabel(r'$x(L_0)$', usetex=True)
plt.grid(True)
plt.title(r'Effect of High Frequency Driving', usetex=True)
plt.legend()
plt.savefig('Dimensionless_High Frequency Force.pdf')
plt.show()

# %%
zeta = 0.5  # Damping factor for the underdamped case
t_values_underdamped, x_values_underdamped, noise_iso_underdamped = RK45(f, tau0, tauf, S0, h)
zeta = 1  # Damping factor for the critically damped case
t_values_criticallydamped, x_values_criticallydamped, noise_iso_criticallydamped = RK45(f, tau0, tauf, S0, h)
zeta = 1.5  # Damping factor for the overdamped case
t_values_overdamped, x_values_overdamped, noise_iso_overdamped = RK45(f, tau0, tauf, S0, h)

# Plot the results for different damping factors
plt.plot(t_values_underdamped, x_values_underdamped[:, 0], label='Underdamped')
plt.plot(t_values_criticallydamped, x_values_criticallydamped[:, 0], label='Critically Damped')
plt.plot(t_values_overdamped, x_values_overdamped[:, 0], label='Overdamped')
plt.xlabel(r'$\tau(\frac{1}{\omega_m})$', usetex=True)
plt.ylabel(r'$x(L_0)$', usetex=True)
plt.grid(True)
plt.title(r'Different Damping', usetex=True)
plt.legend()
plt.savefig('Dimensionless_Different damping.pdf')
plt.show()
