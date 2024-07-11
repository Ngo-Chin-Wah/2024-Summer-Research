#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:35:36 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def F(t):    
    return F0 * np.cos(omega_f * t)


def f(noise_temp, noisiness_temp, t, S):
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = F(t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness_temp * noise_temp
    return dSdt

def RK45(noisiness_temp, f, t0, tf, S0, h):
    t_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    n = 0
    noise_iso = np.empty(0, dtype=float)

    while t < tf:
        noise_temp = np.random.normal(loc=0, scale=1)
        noise_iso = np.append(noise_iso, noise_temp)
        n = n + 1
        x = x_values[-1, :]
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
            h = s * h
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

        x_values = np.concatenate((x_values, [x_new]), axis=0)
        t_values = np.append(t_values, t + h)
        t = t + h
    return t_values, x_values, noise_iso
# %%

global r, omega, error_m, omega_f, F0, h_interpolate, noisiness

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
T = 297
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-5
F0 = 1
# %%
noisiness = 0
omega_f = np.sqrt(6)

t_values_spline = np.arange(t0, tf, h_interpolate)
force = np.empty(0, dtype=float)

for t in t_values_spline:
    force = np.append(force, F(t))
    
t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)

plt.plot(t_values_spline, x_values_spline, label='Oscillator')
plt.plot(t_values_spline, force, label='Driving Force')
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'When $\omega = \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('xt_Resonance.pdf')
plt.show()

omega_f = np.sqrt(4)

t_values_spline = np.arange(t0, tf, h_interpolate)
force = np.empty(0, dtype=float)

for t in t_values_spline:
    force = np.append(force, F(t))
    
t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)

plt.plot(t_values_spline, x_values_spline, label='Oscillator')
plt.plot(t_values_spline, force, label='Driving Force')
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'When $\omega < \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('xt_Below Resonance.pdf')
plt.show()

omega_f = np.sqrt(8)

t_values_spline = np.arange(t0, tf, h_interpolate)
force = np.empty(0, dtype=float)

for t in t_values_spline:
    force = np.append(force, F(t))
    
t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)

plt.plot(t_values_spline, x_values_spline, label='Oscillator')
plt.plot(t_values_spline, force, label='Driving Force')
plt.xlabel('t')
plt.ylabel('x')
plt.title(r'When $\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('xt_Above Resonance.pdf')
plt.show()

# %%

omega_f = np.sqrt(6)
noisiness = 0
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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs without Noise when $\omega = \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0_Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs without Noise when $\omega = \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0 Semi Log_Resonance.pdf')
plt.show()

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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega = \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43_Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega = \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43 Semi Log_Resonance.pdf')
plt.show()
# %%

omega_f = np.sqrt(4)
noisiness = 0
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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs without Noise when $\omega < \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0_Below Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs without Noise when $\omega < \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0 Semi Log_Below Resonance.pdf')
plt.show()

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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega < \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43_Below Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega < \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43 Semi Log_Below Resonance.pdf')
plt.show()
# %%

omega_f = np.sqrt(8)
noisiness = 0
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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs without Noise when $\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0_Above Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs without Noise when $\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 0 Semi Log_Above Resonance.pdf')
plt.show()

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

plt.plot(t_values_spline50, radius_mean50)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43_Above Resonance.pdf')
plt.show()

plt.plot(t_values_spline50, np.log(radius_mean50))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title(r'100 Runs of Noise Strength 43.0 when $\omega > \sqrt{\frac{k}{m}}$', usetex=False)
plt.grid(True)
plt.savefig('Noise Strength 43 Semi Log_Above Resonance.pdf')
plt.show()