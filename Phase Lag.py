#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:55:52 2024

@author: nathanngo
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import pandas as pd

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
        print(t, h)

        while (error > error_m) or (error / error_m < 0.05):
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
            print(t, h)

        x_values = np.concatenate((x_values, [x_new]), axis=0)
        t_values = np.append(t_values, t + h)
        t = t + h
    return t_values, x_values, noise_iso
# %%

global r, omega, error_m, omega_f, F0, h_interpolate, noisiness

m = 0.5
k = 3
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
error_m = 1e-4
F0 = 1
noisiness = 0
# %%

linewidth = gamma / m
freqs = np.arange(omega - 8 * linewidth, omega + 8 * linewidth, 0.01 * linewidth)
t_values_spline = np.arange(t0, tf, h_interpolate)
phase_lags = np.empty(0, dtype=float)

for freq in freqs:
    omega_f = freq
    force = np.empty(0, dtype=float)

    for t in t_values_spline:
        force = np.append(force, F(t))
    
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    
    fft_force = fft(force)
    fft_position = fft(x_values_spline)

    frequencies = fftfreq(len(t_values_spline), d=h_interpolate)
    dominant_index = np.argmax(np.abs(fft_force))

    phase_force = np.angle(fft_force[dominant_index])
    phase_position = np.angle(fft_position[dominant_index])

    phase_lag = phase_position - phase_force
    phase_lag = np.degrees(np.abs(np.mod(phase_lag + np.pi, 2 * np.pi) - np.pi))
    print(freq, phase_lag)
    
    phase_lags = np.append(phase_lags, phase_lag)

plt.plot(freqs, phase_lags)
plt.xlabel('Angular Frequency of the Driving Force')
plt.ylabel('Absolute Value of Phase Lag')
plt.title(r'Natural Frequency = $\sqrt{3}$; Linewidth = $0.1$')
plt.grid(True)
plt.savefig('Phase Lag.pdf')
plt.show()
# %%

# from scipy.ndimage import gaussian_filter1d
# smooth = gaussian_filter1d(phase_lags, 10)

# # compute second derivative
# smooth_d2 = np.gradient(np.gradient(smooth))

# # find switching points
# infls = np.where(np.diff(np.sign(smooth_d2)))[0]
# print(freqs[infls])

# # plot results
# plt.plot(freqs, phase_lags, label='Noisy Data')
# plt.plot(freqs, smooth, label='Smoothed Data')
# plt.legend()
# plt.show()
# %%

window_size = 1
data = pd.DataFrame({'Frequency': freqs, 'Phase Lag': phase_lags})
data['Filtered'] = data['Phase Lag'].rolling(window=window_size).mean()

data['dy'] = np.gradient(data['Filtered'], data['Frequency'])
data['ddy'] = np.gradient(data['Filtered'], data['Frequency'])

threshold = 1.0
data['significant_ddy'] = np.where(np.abs(data['ddy']) > threshold, data['ddy'], 0)
data['sign_change'] = np.sign(data['significant_ddy']).diff().ne(0).astype(int)
inflection_points =data[data['sign_change'] == 1]

print("Inflection points:")
print(inflection_points[['Frequency', 'Filtered']])

plt.plot(data['Frequency'], data['Filtered'], label='Original Data')
plt.scatter(inflection_points['Frequency'], inflection_points['Filtered'], color='red', zorder=5, label='Inflection Points')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title(r'Natural Frequency = $\sqrt{6}$; Linewidth = $0.2$')
plt.savefig('Inflection Points.pdf')
plt.show()

# %%

# Filter inflection points between frequency 2 and 3
filtered_inflection_points = inflection_points[(inflection_points['Frequency'] >= 3.5) & (inflection_points['Frequency'] <= 4.5)]

# Print the inflection points one by one
print("Inflection points between frequency 2 and 3:")
for index, row in filtered_inflection_points.iterrows():
    print(f"Index: {index}, Frequency: {row['Frequency']}, Filtered: {row['Filtered']}")