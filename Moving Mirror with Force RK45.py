import numpy as np
import matplotlib.pyplot as plt
from vpython import *

scene = canvas(width = 1500, height = 400, background = color.white)

def F(t):
    return F0 * np.cos(omega_f * t) 

def f(t, S):
    """
    Defines the function representing the system of differential equations.

    Parameters:
        t (float): Time variable.
        S (numpy.ndarray): State vector [position, velocity].

    Returns:
        numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)
    dSdt[0] = S[1]
    dSdt[1] = - F(t) / m - 2 * r * S[1] - omega ** 2 * S[0]
    return dSdt

def RK45(f, t0, tf, S0, h):
    """
    Implements the Runge-Kutta-Fehlberg method (RK45) for solving ordinary differential equations.

    Parameters:
        f (function): Function defining the system of differential equations.
        t0 (float): Initial time.
        tf (float): Final time.
        S0 (numpy.ndarray): Initial state vector [position, velocity].
        h (float): Step size.

    Returns:
        numpy.ndarray: Array of time values.
        numpy.ndarray: Array of state vectors.
    """
    t_values = np.array([t0])
    x_values = np.array([[S0[0], S0[1]]])
    t = t0
    n = 0

    while t < tf:
        n = n + 1
        x = x_values[-1, :]
        k1 = h * f(t, x)
        k2 = h * f(t + (1 / 4) * h, x + (1 / 4) * k1)
        k3 = h * f(t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
        k4 = h * f(t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
        k5 = h * f(t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
        k6 = h * f(t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
        error = abs(z_new[0] - x_new[0])
        s = 0.84 * (error_m / error) ** (1 / 4)
        print("Out loop", n, "h:", h)

        while (error > error_m):
            h = s * h
            k1 = h * f(t, x)
            k2 = h * f(t + (1 / 4) * h, x + (1 / 4) * k1)
            k3 = h * f(t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6
            error = abs(z_new[0] - x_new[0])
            s = 0.84 * (error_m / error) ** (1 / 4)
            print("In loop, h:", h)

        x_values = np.concatenate((x_values, [x_new]), axis = 0)
        t_values = np.append(t_values, t + h)
        t = t + h
    return t_values, x_values

def update_mirror(mirror, sol):
    """
    Updates the position of the mirror.

    Parameters:
        mirror (vpython.box): The mirror object.
        sol (numpy.ndarray): Solution array containing position and velocity.
    """
    mirror.pos = vector(sol[:, 0][-1], 0, 0)

global r, omega, error_m, omega_f, F0

m = float(input("Mass of the mirror:"))
k = float(input("Spring constant:"))
gamma = float(input("gamma:"))
r = gamma / (2 * m)
omega = (k / m) ** 0.5
x0 = float(input("Initial position:"))
v0 = float(input("Initial velocity:"))
t0 = 0.0
tf = 10.0
h = 0.1
S0 = np.array([x0, v0])
error_m = 1e-7
F0 = 0
omega_f = sqrt(6) #The resonance frequency is sqrt(k / m)

mirror = box(pos = vector(S0[0], 0, 0), size = vector(0.01, 0.5, 0.3), color = color.blue)

t_values, x_values = RK45(f, t0, tf, S0, h)

plt.plot(t_values, x_values[:, 0], label = 'Position')
plt.plot(t_values, x_values[:, 1], label = 'Velocity')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position and Velocity against Time')
plt.legend()
plt.grid(True)
plt.savefig('moving_mirror_with_force_rk45.png')

while True:
    for i in range(len(t_values)):
        if (i > 1):
            time_step = t_values[i] - t_values[i - 1]
            fps = 1 / time_step
            rate(fps)
            update_mirror(mirror, x_values[:i + 1])
