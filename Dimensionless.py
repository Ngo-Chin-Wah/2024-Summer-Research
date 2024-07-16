#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 11 15:34:47 2024

@author: nathanngo

Attributes:
    error_m (float): Description
    F0 (int): Description
    h (float): Initial size of time-step in RK45
    h_interpolate (float): Desired size of time-step in interpolation
    noisiness (int): Amplitude of stochastic noise
    noisiness_f (int): Amplitude of stochastic phase shift
    omega_f (int): Driving frequency
    S0 (TYPE): Description
    tau0 (float): Initial time
    tauf (float): Final time
    v0 (float): Initial velocity
    x0 (float): Initial position
    zeta (float): Damping ratio
"""

# %%

import numpy as np
import matplotlib.pyplot as plt


def F(noise_f_temp, tau):
    """Summary
    
    Args:
        noise_f_temp (TYPE): Description
        tau (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return F0 * np.cos(omega_f * tau + noisiness_f * noise_f_temp)


def f(noise_f_temp, noise_temp, tau, S):
    """Summary
    
    Args:
        noise_f_temp (TYPE): Description
        noise_temp (TYPE): Description
        tau (TYPE): Description
        S (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, tau) - 2 * zeta * S[1] - S[0] - noisiness * noise_temp
    return dSdt


def RK45(f, tau0, tauf, S0, h):
    """Summary
    
    Args:
        f (TYPE): Description
        tau0 (TYPE): Description
        tauf (TYPE): Description
        S0 (TYPE): Description
        h (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    tau_values = np.array([tau0])
    x_values = np.array([[S0[0], S0[1]]])
    tau = tau0
    n = 0
    noise_iso = np.empty(0, dtype=float)
    noise_f_iso = np.empty(0, dtype=float)

    while tau < tauf:
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)
        noise_f_temp = np.random.normal(loc=0, scale=1)
        noise_f_iso = np.append(noise_f_iso, noise_f_temp)
        n = n + 1
        x = x_values[-1, :]
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

        x_values = np.concatenate((x_values, [x_new]), axis=0)
        tau_values = np.append(tau_values, tau + h)
        tau = tau + h
    return tau_values, x_values, noise_iso
# %%

global error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

x0 = 0.0
v0 = 3.0
tau0 = 0.0
tauf = 70.0
S0 = np.array([x0, v0])

h = 0.1
h_interpolate = 0.01
error_m = 1e-6

F0 = 0
omega_f = 100

noisiness = 0
noisiness_f = 0
# %%
zeta = 0.5
t_values_underdamped, x_values_underdamped, noise_iso_underdamped = RK45(f, tau0, tauf, S0, h)
zeta = 1
t_values_criticallydamped, x_values_criticallydamped, noise_iso_criticallydamped = RK45(f, tau0, tauf, S0, h)
zeta = 1.5
t_values_overdamped, x_values_overdamped, noise_iso_overdamped = RK45(f, tau0, tauf, S0, h)

plt.plot(t_values_underdamped, x_values_underdamped[:, 0], label='Underdamped')
plt.plot(t_values_criticallydamped, x_values_criticallydamped[:, 0], label='Critically Damped')
plt.plot(t_values_overdamped, x_values_overdamped[:, 0], label='Overdamped')
plt.xlabel(r'$\tau(\frac{1}{\omega_m})$', usetex=True)
plt.ylabel(r'$x(L_0)$', usetex=True)
plt.grid(True)
plt.title(r'Overdamped; $r=\omega$', usetex=True)
plt.legend()
plt.savefig('Dimensionless_Overdamped.pdf')
plt.show()
