import numpy as np
import matplotlib.pyplot as plt
from vpython import *
# scene = canvas(width = 1500, height = 400, background = color.white)

def f(t, S, r, omega):
    """
    Define the function for the second-order ordinary differential equation:
        x'' + 2 * r * x' + omega ** 2 * x = 0
    as a system of first-order differential equations:
        S[0] = x
        S[1] = x'

    Parameters:
        t: float - Time
        S: array-like - Array of dependent variable values [x, x']
        r: float - Damping coefficient
        omega: float - Angular frequency

    Returns:
        dSdt: array-like - Array of derivatives [x', x'']
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]  # x' = y[1]
    dSdt[1] = -2 * r * S[1] - (omega ** 2) * S[0]
    return dSdt

def explicit_RK4(f, t0, tf, y0, r, omega, h):
    """
    Solve the second-order ordinary differential equation using the
    explicit fourth-order Runge-Kutta method.

    Parameters:
        f: function(t, y, r, omega) - The function defining the differential equation
        t0: float - Initial value of the independent variable (time)
        tf: float - Final value of the independent variable
        y0: array-like - Initial value of the dependent variable(s) [x0, v0]
        r: float - Damping coefficient
        omega: float - Angular frequency
        h: float - Step size

    Returns:
        t_values: array - Array of time values
        x_values: array - Array of solution values corresponding to each time
    """

    num_steps = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, num_steps + 1)
    x_values = np.zeros((len(t_values), len(S0)))
    x_values[0] = S0

    for i in range(num_steps):
        t = t_values[i]
        x = x_values[i]

        k1 = h * f(t, x, r, omega)
        k2 = h * f(t + 0.5 * h, x + 0.5 * k1, r, omega)
        k3 = h * f(t + 0.5 * h, x + 0.5 * k2, r, omega)
        k4 = h * f(t + h, x + k3, r, omega)

        x_values[i + 1] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, x_values

# Parameters
m = float(input("Mass of the mirror:"))
k = float(input("Spring constant:"))
gamma = float(input("gamma:"))
r = gamma / (2 * m)
omega = (k / m) ** 0.5
x0 = float(input("Initial position:"))
v0 = float(input("Initial velocity:"))
t0 = 0.0  
tf = 5.0
h = 0.00001 
S0 = np.array([x0, v0])

# Solve the differential equation using RK4
t_values, sol = explicit_RK4(f, t0, tf, S0, r, omega, h)

# Function to update the position of the mirror
def update_mirror(mirror, sol):
    mirror.pos = vector(sol[:, 0][-1], 0, 0)

# Create the mirror object
# mirror = box(pos = vector(S0[0], 0, 0), size = vector(0.01, 0.5, 0.3), color = color.blue)

fps = int(1 / h)

# Apply FFT
n = len(t_values)  # Number of samples
fhat = np.fft.fft(sol[:, 0], n)  # Compute the FFT
psd = fhat * np.conj(fhat) / n  # Power spectral density
freq = (1 / (0.00001 * n)) * np.arange(n)  # Frequency array
L = np.arange(1, np.floor(n / 2), dtype = 'int')  # Only use the first half of the FFT output

# Find the peak in the frequency domain
peak_idx = np.argmax(psd[L])  # Index of the peak
peak_freq = freq[L][peak_idx]  # Frequency of the peak
peak_power = psd[L][peak_idx]  # Power at the peak
print("Peak frequency:", peak_freq)

# Plot the solution
plt.plot(t_values[:10 * fps], sol[:10 * fps, 0], label = 'Position')
plt.plot(t_values[:10 * fps], sol[:10 * fps, 1], label = 'Velocity')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position and Velocity against Time')
plt.legend()
plt.grid(True)
plt.savefig('moving_mirror_rk4.png')  # Save the plot to an image file

# Animate the motion of the mirror
# for i in range(len(t_values)):
    # scene.autoscale = False
    # rate(fps)  # Limit the frame rate to 100 frames per second
    # update_mirror(mirror, sol[:i + 1])