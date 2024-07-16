#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:09:02 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def F(noise_f_temp, t):    
    return F0 * np.cos(omega_f * t + noisiness_f * noise_f_temp)


def f(noise_f_temp, noise_temp, t, S):
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(noise_f_temp, t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness * noise_temp
    return dSdt

def RK45(f, t0, tf, S0, h):
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
        n = n + 1
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
        print(t, h)

        while (error > error_m):
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
            print(t, h)

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
tf = 100.0
h = 0.1
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-6
F0 = 1
noisiness = 0
noisiness_f = 0
omega_f = 100

# %%
t_values, x_values, noise_iso = RK45(f, t0, tf, S0, h)

plt.plot(t_values, x_values)
plt.show()

# %%
noisiness_f = 0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)
for i in range(1):
    t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    # print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

noisiness_f = 1
radius_spline_ps = np.empty(len(t_values_spline), dtype=float)
for i in range(1):
    t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline_ps = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline_ps = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline_ps) ** 2 + (m * v_values_spline_ps) ** 2)
    radius_spline_ps = np.vstack([radius_spline_ps, radius_temp])
    # print(i)
radius_spline_ps = radius_spline_ps[1:]
radius_mean_ps = np.mean(radius_spline_ps, axis=0)

plt.plot(t_values_spline, v_values_spline, label='Deterministic Force')
plt.plot(t_values_spline, v_values_spline_ps, label='Stochastic Force')
plt.xlabel('t')
plt.ylabel('v')
plt.title(r'$\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.legend()
plt.savefig('Stochastic_Above Resonance.pdf')
plt.show()

plt.plot(x_values_spline, m * v_values_spline, label='Deterministic Force')
plt.plot(x_values_spline_ps, m * v_values_spline_ps, label='Stochastic Force')
plt.xlabel('x')
plt.ylabel('p')
plt.grid(True)
plt.legend()
plt.title(r'$\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.savefig('Stoachstic Force Phase Space_Above Resonance.pdf')
plt.show()

plt.plot(t_values_spline, radius_mean, label='Deterministic Force')
plt.plot(t_values_spline, radius_mean_ps, label='Stochastic Force')
plt.xlabel('t')
plt.ylabel('r')
plt.title(r'$\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.legend()
plt.savefig('Stochastic Force Radius_Above Resonance.pdf')
plt.show()