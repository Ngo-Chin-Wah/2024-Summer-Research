#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 11 15:34:47 2024

@author: nathanngo

Attributes:
    error_m (float): Description
    F0 (int): Description
    h (float): Description
    h_interpolate (float): Description
    noisiness (int): Description
    noisiness_f (int): Description
    omega_f (int): Description
    S0 (TYPE): Description
    tau0 (float): Description
    tauf (float): Description
    v0 (float): Description
    x0 (float): Description
    zeta (float): Description
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

zeta = 0.05

x0 = 0.0
v0 = 3.0
tau0 = 0.0
tauf = 30.0
S0 = np.array([x0, v0])

h = 0.1
h_interpolate = 0.01
error_m = 1e-6

F0 = 1
omega_f = 1

noisiness = 0
noisiness_f = 0
# %%
t_values, x_values, noise_iso = RK45(f, tau0, tauf, S0, h)

plt.plot(t_values, x_values)
plt.xlabel(r'$\tau(\frac{1}{\omega_m})$', usetex=True)
plt.ylabel(r'$x(L_0)$', usetex=True)
plt.grid(True)
plt.show()
