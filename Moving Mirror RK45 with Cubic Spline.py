import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from vpython import *  # Importing vpython for visualization

# scene = canvas(width = 1500, height = 400, background = color.white)  # Creating a canvas for visualization

def F(t):
    """
    Calculates the driving force at time t.

    Parameters:
        t (float): Time variable.

    Returns:
        float: The driving force at time t.
    """
    return F0 * np.cos(omega_f * t)  # Driving force as a function of time

def f(t, S):
    """
    Defines the function representing the system of differential equations.

    Parameters:
        t (float): Time variable.
        S (numpy.ndarray): State vector [position, velocity].

    Returns:
        numpy.ndarray: Derivatives of the state vector.
    """
    dSdt = np.zeros_like(S)  # Initialize derivative vector
    dSdt[0] = S[1]  # Derivative of position is velocity
    dSdt[1] = - F(t) / m - 2 * r * S[1] - omega ** 2 * S[0]  # Derivative of velocity (Newton's second law)
    return dSdt  # Return the derivatives

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
    t_values = np.array([t0])  # Array of time values
    x_values = np.array([[S0[0], S0[1]]])  # Array of state vectors
    t = t0  # Initialize time variable
    n = 0  # Initialize iteration counter

    while t < tf:  # Loop until final time
        n = n + 1  # Increment iteration counter
        x = x_values[-1, :]  # Current state vector
        k1 = h * f(t, x)  # Calculate k1
        k2 = h * f(t + (1 / 4) * h, x + (1 / 4) * k1)  # Calculate k2
        k3 = h * f(t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)  # Calculate k3
        k4 = h * f(t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)  # Calculate k4
        k5 = h * f(t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)  # Calculate k5
        k6 = h * f(t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)  # Calculate k6
        x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5  # Update state vector
        z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6  # Calculate z_new for error estimation
        error = abs(z_new[0] - x_new[0])  # Calculate error
        s = 0.84 * (error_m / error) ** (1 / 4)  # Calculate scaling factor
        print("Out loop", n, "h:", h)  # Print debug information

        while (error > error_m):  # Error control loop
            h = s * h  # Adjust step size
            k1 = h * f(t, x)  # Recalculate k1
            k2 = h * f(t + (1 / 4) * h, x + (1 / 4) * k1)  # Recalculate k2
            k3 = h * f(t + (3 / 8) * h, x + (3 / 32) * k1 + (9 / 32) * k2)  # Recalculate k3
            k4 = h * f(t + (12 / 13) * h, x + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)  # Recalculate k4
            k5 = h * f(t + h, x + (439 / 216) * k1 - 8 * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)  # Recalculate k5
            k6 = h * f(t + (1 / 2) * h, x - (8 / 27) * k1 + 2 * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)  # Recalculate k6
            x_new = x + (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4101) * k4 - (1 / 5) * k5  # Update state vector
            z_new = x + (16 / 135) * k1 + (6656 / 12825) * k3 + (28561 / 56430) * k4 - (9 / 50) * k5 + (2 / 55) * k6  # Calculate z_new for error estimation
            error = abs(z_new[0] - x_new[0])  # Calculate error
            s = 0.84 * (error_m / error) ** (1 / 4)  # Calculate scaling factor
            print("In loop, h:", h)  # Print debug information

        x_values = np.concatenate((x_values, [x_new]), axis = 0)  # Append new state to the array
        t_values = np.append(t_values, t + h)  # Append new time to the array
        t = t + h  # Update time variable
    return t_values, x_values  # Return time and state arrays

# def update_mirror(mirror, sol):
"""
    Updates the position of the mirror.

    Parameters:
        mirror (vpython.box): The mirror object.
        sol (numpy.ndarray): Solution array containing position and velocity.
    """
    # mirror.pos = vector(sol[:, 0][-1], 0, 0)  # Update mirror position based on the solution

def cubic_spline_coefficients(t_values, x_values):
    """
    Computes the coefficients for the cubic spline interpolation.

    Parameters:
        t_values (numpy.ndarray): Array of time values.
        x_values (numpy.ndarray): Array of position values.

    Returns:
        tuple: Coefficients (a, b, c, d) for the cubic spline.
    """
    n = len(t_values) - 1  # Number of intervals
    h = np.diff(t_values)  # Interval sizes
    b = np.diff(t_values) / h  # Slope between points
    
    # Set up the tridiagonal system
    A = np.zeros((n + 1, n + 1))  # Matrix A
    rhs = np.zeros(n + 1)  # Right-hand side vector
    
    A[0, 0] = 1  # Boundary condition at the start
    A[n, n] = 1  # Boundary condition at the end
    
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]  # Lower diagonal
        A[i, i] = 2 * (h[i - 1] + h[i])  # Main diagonal
        A[i, i + 1] = h[i]  # Upper diagonal
        rhs[i] = 3 * (b[i] - b[i - 1])  # Right-hand side
        print("Cubic splining", i)  # Print debug information
    
    # Solve the system for c
    c = np.linalg.solve(A, rhs)  # Solve the tridiagonal system
    
    # Calculate the spline coefficients
    a = x_values[:-1]  # Coefficient a
    b = b - (h * (2 * c[:-1] + c[1:])) / 3  # Coefficient b
    d = (c[1:] - c[:-1]) / (3 * h)  # Coefficient d
    
    return a, b, c[:-1], d  # Return the coefficients

def cubic_spline(t_values, x_values, t_values_splined):
    """
    Performs cubic spline interpolation on the given data.

    Parameters:
        t_values (numpy.ndarray): Array of time values.
        x_values (numpy.ndarray): Array of position values.
        t_values_splined (numpy.ndarray): Array of time values for the spline.

    Returns:
        numpy.ndarray: Interpolated position values.
    """
    a, b, c, d = cubic_spline_coefficients(t_values, x_values)  # Get spline coefficients
    n = len(t_values) - 1  # Number of intervals
    
    def spline(x):
        for i in range(n):
            if t_values[i] <= x <= t_values[i + 1]:  # Check which interval x is in
                dx = x - t_values[i]  # Compute the distance from the interval start
                return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3  # Compute spline value
        return None

    return np.array([spline(x) for x in t_values_splined])  # Compute spline for all t_values_splined

# Global variables
global r, omega, error_m, omega_f, F0

# Input parameters
m = float(input("Mass of the mirror:"))  # Mass of the mirror
k = float(input("Spring constant:"))  # Spring constant
gamma = float(input("gamma:"))  # Damping coefficient

# Derived parameters
r = gamma / (2 * m)  # Damping ratio
omega = (k / m) ** 0.5  # Natural frequency

# Initial conditions
x0 = float(input("Initial position:"))  # Initial position
v0 = float(input("Initial velocity:"))  # Initial velocity
t0 = 0.0  # Start time
tf = 40.0  # End time
h = 0.1  # Initial step size
S0 = np.array([x0, v0])  # Initial state vector
error_m = 1e-5  # Tolerance for error
F0 = 0  # Driving force amplitude
omega_f = np.sqrt(6)  # Driving frequency

# mirror = box(pos = vector(S0[0], 0, 0), size = vector(0.01, 0.5, 0.3), color = color.blue)  # Create a mirror object

# Solving the ODE using RK45 method
t_values, x_values = RK45(f, t0, tf, S0, h)  # Solve the differential equations

# Cubic spline interpolation
t_values_splined = np.linspace(t0, tf, int((tf - t0) / 0.001))  # Generate dense time values for spline
x_values_splined = cubic_spline(t_values, x_values[:, 0], t_values_splined)  # Compute spline values
# print("t_values_splined:", t_values_splined)  # Print spline time values
# print("x_values_splined", x_values_splined)  # Print spline position values

# Apply FFT
n = len(t_values_splined)  # Number of samples
fhat = np.fft.fft(x_values_splined, n)  # Compute the FFT
psd = fhat * np.conj(fhat) / n  # Power spectral density
freq = (1 / (0.001 * n)) * np.arange(n)  # Frequency array
L = np.arange(1, np.floor(n / 2), dtype = 'int')  # Only use the first half of the FFT output

# Find the peak in the frequency domain
peak_idx = np.argmax(psd[L])  # Index of the peak
peak_freq = freq[L][peak_idx]  # Frequency of the peak
peak_power = psd[L][peak_idx]  # Power at the peak

# Plotting the results
plt.figure(figsize = (10, 8))  # Create a figure

plt.subplot(5, 1, 1)  # Create a subplot for position and velocity
plt.plot(t_values, x_values[:, 0], label = 'Position')  # Plot position
plt.plot(t_values, x_values[:, 1], label = 'Velocity')  # Plot velocity
plt.xlabel('Time')  # X-axis label
plt.ylabel('Position')  # Y-axis label
plt.title('Position and Velocity against Time')  # Plot title
plt.legend()  # Show legend
plt.grid(True)  # Show grid

plt.subplot(5, 1, 3)  # Create a subplot for the cubic spline
plt.plot(t_values_splined, x_values_splined)  # Plot spline values
plt.xlabel('Time')  # X-axis label
plt.ylabel('Position')  # Y-axis label
plt.title('Position against Time (Cubic Spline)')  # Plot title
plt.grid(True)  # Show grid

plt.subplot(5, 1, 5)  # Create a subplot for the FFT
plt.plot(freq[L], psd[L])  # Plot the power spectrum
plt.text(peak_freq, peak_power, f'({peak_freq:.01f}, {peak_power:.01f})', color = 'black', ha = "right")  # Annotate the peak
plt.xlabel('Frequency (Hz)')  # X-axis label
plt.ylabel('Power')  # Y-axis label
plt.title('Frequency Domain Signal')  # Plot title
plt.grid(True)  # Show grid
plt.autoscale()
plt.savefig('moving_mirror_with_force_rk45_cubic_spline.png')  # Save the plot

# while True:
#     for i in range(len(t_values)):
#         if (i > 1):
#             time_step = t_values[i] - t_values[i - 1]
#             fps = 1 / time_step
#             rate(fps)
#             update_mirror(mirror, x_values[:i + 1])