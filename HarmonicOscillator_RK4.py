import numpy as np
import matplotlib.pyplot as plt
from vpython import *
# scene = canvas(width = 1500, height = 400, background = color.white)

def f(t, S, r, omega):
    """
    Defines the system of first-order differential equations from the original second-order ODE:
        x'' + 2 * r * x' + omega ** 2 * x = 0

    Parameters:
        t (float): Time variable.
        S (array-like): Array containing [x, x'], where x is position and x' is velocity.
        r (float): Damping coefficient.
        omega (float): Angular frequency.

    Returns:
        array-like: Array of derivatives [x', x''].
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]  # x' = S[1]
    dSdt[1] = -2 * r * S[1] - (omega ** 2) * S[0]  # x'' = -2r * x' - omega^2 * x
    return dSdt

def explicit_RK4(f, t0, tf, y0, r, omega, h):
    """
    Solves a system of differential equations using the fourth-order Runge-Kutta (RK4) method.

    Parameters:
        f (function): The function defining the differential equation.
        t0 (float): Initial time.
        tf (float): Final time.
        y0 (array-like): Initial values [x0, v0].
        r (float): Damping coefficient.
        omega (float): Angular frequency.
        h (float): Step size.

    Returns:
        array: Array of time values.
        array: Array of solution values for each time step.
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
r = gamma / (2 * m)  # Damping coefficient
omega = (k / m) ** 0.5  # Angular frequency
x0 = float(input("Initial position:"))
v0 = float(input("Initial velocity:"))
t0 = 0.0  # Initial time
tf = 5.0  # Final time
h = 0.00001  # Step size
S0 = np.array([x0, v0])  # Initial state vector [position, velocity]

# Solve the differential equation using RK4
t_values, sol = explicit_RK4(f, t0, tf, S0, r, omega, h)

# Function to update the position of the mirror
def update_mirror(mirror, sol):
    """
    Updates the position of the mirror in the vpython simulation.

    Parameters:
        mirror (vpython.box): The mirror object in the vpython scene.
        sol (array): Solution array containing the position and velocity.
    """
    mirror.pos = vector(sol[:, 0][-1], 0, 0)

# Create the mirror object (uncomment to use vpython visualization)
# mirror = box(pos = vector(S0[0], 0, 0), size = vector(0.01, 0.5, 0.3), color = color.blue)

fps = int(1 / h)  # Frames per second based on the step size

# Apply FFT to analyze the frequency domain
n = len(t_values)  # Number of samples
fhat = np.fft.fft(sol[:, 0], n)  # Compute the FFT of the position data
psd = fhat * np.conj(fhat) / n  # Power spectral density
freq = (1 / (0.00001 * n)) * np.arange(n)  # Frequency array
L = np.arange(1, np.floor(n / 2), dtype='int')  # Only use the first half of the FFT output

# Find the peak in the frequency domain
peak_idx = np.argmax(psd[L])  # Index of the peak
peak_freq = freq[L][peak_idx]  # Frequency of the peak
peak_power = psd[L][peak_idx]  # Power at the peak
print("Peak frequency:", peak_freq)

# Plot the solution (Position and Velocity against Time)
plt.plot(t_values[:10 * fps], sol[:10 * fps, 0], label='Position')
plt.plot(t_values[:10 * fps], sol[:10 * fps, 1], label='Velocity')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position and Velocity against Time')
plt.legend()
plt.grid(True)
plt.savefig('moving_mirror_rk4.png')  # Save the plot to an image file

# Animate the motion of the mirror (uncomment to use vpython visualization)
# for i in range(len(t_values)):
    # scene.autoscale = False
    # rate(fps)  # Limit the frame rate to 100 frames per second
    # update_mirror(mirror, sol[:i + 1])
