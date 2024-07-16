#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:49:24 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

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

def exp_decay(t, A, T1):
    return A * np.exp(-t / T1)
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
h_interpolate = 0.001
S0 = np.array([x0, v0])
error_m = 1e-6
F0 = 0
noisiness = 0
noisiness_f = 1
omega_f = np.sqrt(k / m)
# %%

t_values_spline = np.arange(t0, tf, h_interpolate)

t_values_temp, x_values_temp, noise_iso = RK45(f, t0, tf, S0, h)
interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
x_values_spline = interpolator(t_values_spline)

plt.figure()
plt.plot(t_values_spline, x_values_spline)
plt.xlabel(r'$t$', usetex=True)
plt.ylabel(r'$x$', usetex=True)
plt.title(r'$\omega_m>\sqrt{\frac{k}{m}}$', usetex=True)
plt.savefig('Relaxation Decoherence xt.pdf')
plt.show()
# %%

X = fft(x_values_spline)
freqs = fftfreq(len(t_values_spline), h_interpolate)

X = X[freqs >= 0]
freqs = freqs[freqs >= 0]

amplitude_envelope = np.abs(X)
popt, _ = curve_fit(exp_decay, freqs, amplitude_envelope, p0=[1.0, 1.0])

T1 = popt[1]

peak_freq = freqs[np.argmax(np.abs(X))]
peak_x = np.max(np.abs(X))
print(peak_x)
freq_fwhm = np.empty(0, dtype=float)

for i in range(len(X)):
    if (np.abs(X[i]) >= 0.5 * peak_x):
        freq_fwhm = np.append(freq_fwhm, freqs[i])
print(freq_fwhm)
fwhm = freq_fwhm[-1] - freq_fwhm[0]
T2 = 1 / (np.pi * fwhm)

print('Relaxation Time T1:', T1)
print('Dephasing Time T2:', T2)

plt.plot(freqs[:300], np.abs(X)[:300])
plt.xlabel(r'Frequency', usetex=True)
plt.ylabel(r'Amplitude', usetex=True)
plt.title(r'Fourier Transform', usetex=True)
plt.grid(True)
plt.savefig('Relaxation Decoherence FFT.pdf')
plt.show()
# %%
freqs_new = freqs[:100]
abs_new = np.abs(X)[:100]
freqs_spline = np.arange(np.min(freqs_new), np.max(freqs_new), 0.001)

interpolator = interp1d(freqs_new, abs_new, kind='cubic')
abs_spline = interpolator(freqs_spline)

plt.plot(freqs_spline, abs_spline)
plt.show()
# %%

abs_max = np.max(abs_spline)
freqs_above = np.empty(0, dtype=float)

for i in range(len(freqs_spline)):
    if (abs_spline[i] >= 0.5 * abs_max):
        print(freqs_spline[i])
        freqs_above = np.append(freqs_above, freqs_spline[i])

T2 = freqs_above[-1] - freqs_above[0]
print('T2:', T2)