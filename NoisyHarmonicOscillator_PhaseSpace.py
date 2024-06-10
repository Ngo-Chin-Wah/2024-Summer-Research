import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def F(t):
    """
    Calculates the driving force at time t.

    Parameters:
        t (float): Time variable.

    Returns:
        float: The driving force at time t.
    """
    return F0 * np.cos(omega_f * t)


def f(noise_temp, noisiness_temp, t, S):
    """
    Defines the function representing the system of differential equations.

    Parameters:
        noisiness_temp: noisiness parameter
        noise_temp: Noise variable
        t (float): Time variable.
        S (numpy.ndarray): State vector [position, velocity].

    Returns:
        numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = - F(t) / m - 2 * r * S[1] - (omega ** 2) * S[0] - noisiness_temp * noise_temp
    return dSdt


def RK45(noisiness_temp, f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta-Fehlberg method (RK45) for solving ordinary differential equations.

    Parameters:
        noisiness_temp: Noisiness parameter
        f (function): Function defining the system of differential equations.
        t0 (float): Initial time.
        tf (float): Final time.
        S0 (numpy.ndarray): Initial state vector [position, velocity].
        h (float): Step size.

    Returns:≠6≠
        numpy.ndarray: Array of time values.
        numpy.ndarray: Array of state vectors.
    """
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


global r, omega, error_m, omega_f, F0, h_interpolate, noisiness

m = 0.5
k = 3.0
gamma = 0.1

r = gamma / (2 * m)
omega = (k / m) ** 0.5

x0 = 0.0
v0 = 3.0
t0 = 0.0
tf = 10.0
h = 0.1
T = 297
h_interpolate = 0.01
S0 = np.array([x0, v0])
error_m = 1e-4
F0 = 0
omega_f = np.sqrt(6)
n = 30
radius_mean_mean_y = np.empty(0, dtype=float)

noisiness = np.arange(6, 11, 0.01)

for noisiness_temp in noisiness:
    radius_mean_mean = 0
    for i in range(n):
        radius = np.empty(0, dtype=float)
        t_values, x_values, noise_iso = RK45(noisiness_temp, f, t0, tf, S0, h)
        peaks, _ = find_peaks(x_values[:, 0])
        for peak in peaks:
            radius = np.append(radius, np.abs(x_values[peak][0]))
        radius_mean = np.mean(radius)
        radius_mean_mean += radius_mean
    radius_mean_mean = radius_mean_mean / n
    radius_mean_mean_y = np.append(radius_mean_mean_y, radius_mean_mean)
    print("noisiness:", noisiness_temp, "radius_mean_mean:", radius_mean_mean)

radius_filtered = savgol_filter(radius_mean_mean_y, 101, 2)

plt.plot(noisiness, radius_filtered)
plt.xlabel('Noise Amplitude')
plt.ylabel('Radius')
plt.title('Radius against Noise Amplitude')
plt.grid(True)
plt.savefig('NoisyHarmonicOscillator_PhaseSpace.pdf')
