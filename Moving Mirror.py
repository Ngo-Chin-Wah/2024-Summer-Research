import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from vpython import *
scene = canvas(width=1500, height=400, background = color.white)

# Define the function representing the ODE, assuming F(t) and noise term are 0
def moving_mirror(x, t, r, omega):
    dxdt = [x[1], -2 * r * x[1] - omega ** 2 * x[0]]
    return dxdt

# Parameters
m = float(input("Mass of the mirror:"))
k = float(input("Spring constant:"))
gamma = float(input("gamma:"))
r = gamma / (2 * m)
omega = (k / m) ** 0.5
x_0 = float(input("Initial position:"))
v_0 = float(input("Initial velocity:"))

# Initial conditions
Initial = [x_0, v_0]  # Initial position and velocity

# Time points to solve the ODE for
t = np.linspace(0, 100, 10000)

# Solve the ODE
sol = odeint(moving_mirror, Initial, t, args = (r, omega))

# Function to update the position of the mirror
def update_mirror(mirror, sol):
    mirror.pos = vector(sol[:, 0][-1], 0, 0)

# Create the mirror object
mirror = box(pos = vector(Initial[0], 0, 0), size = vector(0.01, 0.5, 0.3), color = color.blue)

# Plot the solution and sapipve it to an image file
plt.plot(t[:1000], sol[:1000, 0], 'b', label = 'Position x(t)')
plt.plot(t[:1000], sol[:1000, 1], 'g', label = "Velocity x'(t)")
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend(loc = 'best')
plt.title('Position and Velocity against Time')
plt.grid()
plt.savefig('moving_mirror.png')  # Save the plot to an image file

# Animate the motion of the mirror
for i in range(len(t)):
    #scene.autoscale = False
    rate(100)  # Limit the frame rate to 100 frames per second
    update_mirror(mirror, sol[:i + 1])