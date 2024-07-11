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
    return F0 * np.cos(omega_f * t + noisiness_f * noise_f_temp)

def f(noise_f_temp, noise_temp, t, S):
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp
    return dSdt

def euler(f, t0, tf, S0, h):
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
        t = t + h

    return t_values, x_values, noise_iso
# %%

global r, omega, error_m, omega_f, F0, h_interpolate, noisiness, noisiness_f

m = 0.5
k = 3.0
gamma = 0.1

r = gamma / (2 * m)
omega = (k / m) ** 0.5

x0 = 0.0
v0 = 3.0
t0 = 0.0
tf = 20.0
h = 0.001
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-4
F0 = 0
noisiness = 43
noisiness_f = 1
omega_f = np.sqrt(k / m)
# %%

t_values, x_values, noise_iso = euler(f, t0, tf, S0, h)

plt.plot(t_values, x_values[:, 0])
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'$\omega = \sqrt{\frac{k}{m}}$; Euler Method', usetex=False)
plt.grid(True)
plt.show()
