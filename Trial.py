import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
omega_0 = 1.0  # Natural frequency
damping_constants = [0.1, 1.0, 2.0]  # Different damping constants
time = np.linspace(0, 50, 5000)  # Extended time array with finer resolution


# Define the differential equation
def damped_oscillator(t, y, gamma, omega_0):
    x, v = y
    dxdt = v
    dvdt = -2 * gamma * v - omega_0 ** 2 * x
    return [dxdt, dvdt]


# Function to compute Fourier transform
def compute_fourier_transform(signal, dt):
    return np.fft.fftshift(np.fft.fft(signal)) * dt


# Time step
dt = time[1] - time[0]

# Initial conditions: x(0) = 1, v(0) = 0
initial_conditions = [1.0, 0.0]

# Prepare plot
plt.figure(figsize=(18, 6))

# Simulate and plot for each damping constant
for i, gamma in enumerate(damping_constants):
    # Solve the differential equation
    sol = solve_ivp(damped_oscillator, [time[0], time[-1]], initial_conditions, t_eval=time, args=(gamma, omega_0))
    x_t = sol.y[0]  # Displacement x(t)

    # Compute Fourier Transform
    freq = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt))
    ft_signal = compute_fourier_transform(x_t, dt)

    # Plotting the Fourier Transform
    plt.subplot(1, 3, i + 1)
    plt.plot(freq, np.abs(ft_signal))
    plt.title(f'Fourier Transform (Damping Constant = {gamma})')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()
