import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the function representing the ODE, assuming F(t) and noise term are 0
def harmonic_oscillatorscipy(x, t, r, omega):
    dxdt = [x[1], -2 * r * x[1] - omega ** 2 * x[0]]
    return dxdt

# Parameters
m = float(input("Mass of the mirror:"))
k = float(input("Spring constant:"))
gamma = float(input("gamma:"))
r = gamma / (2 * m)
omega = (k / m) ** 0.5
x_0 = float(input("Initial position"))
v_0 = float(input("Initial velocity:"))

# Initial conditions
Initial = [x_0, v_0]  # Initial position and velocity

# Time points to solve the ODE for
t = np.linspace(0, 30, 10000)

# Solve the ODE
sol = odeint(harmonic_oscillatorscipy, Initial, t, args = (r, omega))

# Plot the solution and sapipve it to an image file
plt.plot(t, sol[:, 0], 'b', label='Position x(t)')
plt.plot(t, sol[:, 1], 'g', label="Velocity x'(t)")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(loc = 'best')
plt.title('Position and Velocity against Time')
plt.grid()
plt.savefig('moving_mirror.png')  # Save the plot to an image file