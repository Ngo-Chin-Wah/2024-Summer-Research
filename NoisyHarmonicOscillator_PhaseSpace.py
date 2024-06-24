import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def F(t):    
    return F0 * np.cos(omega_f * t)


def f(noise_temp, noisiness_temp, t, S):
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = - F(t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness_temp * noise_temp
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
tf = 100
h = 0.1
T = 297
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-6
F0 = 0
omega_f = np.sqrt(6)
# %%
noisiness = 40
t_values, x_values, noise_iso= RK45(noisiness, f, t0, tf, S0, h)
radius = np.sqrt((x_values[:, 0]) ** 2 + (m * x_values[:, 1]) ** 2)

plt.plot(x_values[:, 0], m * x_values[:, 1])
plt.xlabel('x')
plt.ylabel('p')
plt.title('Noise Strength 42.5; Single Run')
plt.grid(True)
plt.show()

plt.plot(t_values, radius)
plt.xlabel('t')
plt.ylabel('r')
plt.title('Noise Strength 42.5; Single Run')
plt.grid(True)
plt.show()
# %%
noisiness = 20.0
t_values, x_values, noise_iso= RK45(noisiness, f, t0, tf, S0, h)
radius = np.sqrt((x_values[:, 0]) ** 2 + (m * x_values[:, 1]) ** 2)

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(x_values[:, 0], m * x_values[:, 1])
plt.xlabel('x')
plt.ylabel('p')
plt.title('Noise Strength 20.0; Single Run')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_values, radius)
plt.xlabel('t')
plt.ylabel('r')
plt.title('Noise Strength 20.0; Single Run')
plt.grid(True)
plt.show()
# %%
noisiness = 30.0
t_values, x_values, noise_iso= RK45(noisiness, f, t0, tf, S0, h)
radius = np.sqrt((x_values[:, 0]) ** 2 + (m * x_values[:, 1]) ** 2)

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(x_values[:, 0], m * x_values[:, 1])
plt.xlabel('x')
plt.ylabel('p')
plt.title('Noise Strength 30.0; Single Run')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_values, radius)
plt.xlabel('t')
plt.ylabel('r')
plt.title('Noise Strength 30.0; Single Run')
plt.grid(True)
plt.show()
# %%

noisiness = 10.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 10.0; 100 Runs')
plt.grid(True)
plt.show()

plt.plot(np.log(t_values_spline), np.log(radius_mean))
plt.xlabel('ln(t)')
plt.ylabel('ln(<r>)')
plt.title('Noise Strength 10.0; 100 Runs')
plt.grid(True)
plt.show()

plt.plot(t_values_spline, np.log(radius_mean))
plt.xlabel('t')
plt.ylabel('ln(<r>)')
plt.title('Noise Strength 10.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

noisiness = 20.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 20.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

noisiness = 30.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 30.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

noisiness = 40.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 40.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

noisiness = 50.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 50.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

noisiness = 60.0
t_values_spline = np.arange(t0, tf, h_interpolate)
radius_spline = np.empty(len(t_values_spline), dtype=float)

for i in range(100):
    t_values_temp, x_values_temp, noise_iso = RK45(noisiness, f, t0, tf, S0, h)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 0], kind='cubic')
    x_values_spline = interpolator(t_values_spline)
    interpolator = interp1d(t_values_temp, x_values_temp[:, 1], kind='cubic')
    v_values_spline = interpolator(t_values_spline)
    radius_temp = np.sqrt((x_values_spline) ** 2 + (m * v_values_spline) ** 2)
    radius_spline = np.vstack([radius_spline, radius_temp])
    print(i)
radius_spline = radius_spline[1:]
radius_mean = np.mean(radius_spline, axis=0)

plt.plot(t_values_spline, radius_mean)
plt.xlabel('t')
plt.ylabel('<r>')
plt.title('Noise Strength 60.0; 100 Runs')
plt.grid(True)
plt.show()
# %%

n = 100
radius_mean_mean_y = np.empty(0, dtype=float)
noisiness = np.arange(40, 65, 0.05)

for noisiness_temp in noisiness:
    radius_mean_mean = 0
    for i in range(n):
        radius = np.empty(0, dtype=float)
        t_values, x_values, noise_iso = RK45(noisiness_temp, f, t0, tf, S0, h)
        radius = np.sqrt((x_values[:, 0]) ** 2 + (m * x_values[:, 1]) ** 2)
        radius_mean = np.mean(radius)
        radius_mean_mean += radius_mean
        print(i)
    radius_mean_mean = radius_mean_mean / n
    radius_mean_mean_y = np.append(radius_mean_mean_y, radius_mean_mean)
    print("noisiness:", noisiness_temp, "radius_mean_mean:", radius_mean_mean)
    
plt.plot(noisiness, radius_mean_mean_y)
plt.xlabel('Noise Strength')
plt.ylabel('<<r>>')
plt.title('Original Data')
plt.grid(True)
plt.show()
# %%
import pandas as pd
data = pd.DataFrame({'Noise Strength': noisiness, '<<r>>': radius_mean_mean_y})

window_size = 30
data['filtered'] = data['<<r>>'].rolling(window=window_size).mean()

plt.plot(data['Noise Strength'], data['filtered'], label='Smoothed Data')
plt.plot(noisiness, radius_mean_mean_y, label = 'Original Data')
plt.xlabel('Noise Strength')
plt.ylabel('<<r>>')
plt.title('Original and Smoothed Data')
plt.grid(True)
plt.legend()
plt.show()

# %%
plt.plot(data['Noise Strength'], data['filtered'])
plt.xlabel('Noise Strength')
plt.ylabel('<<r>>')
plt.title('Smoothed Data')
plt.grid(True)
plt.show()
# %%

plt.plot(noisiness, radius_mean_mean_y, label = 'Original Data')
plt.plot(data['Noise Strength'], data['filtered'], label='Smoothed Data')
plt.xlabel('Noise Strength')
plt.ylabel('<<r>>')
plt.title('Original and Smoothed Data')
plt.grid(True)
plt.legend()
plt.show()