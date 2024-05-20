import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_points, y_points, x):
    """
    Perform Lagrange interpolation for the given set of data points.

    Parameters:
    x_points (list of float): The x-coordinates of the data points.
    y_points (list of float): The y-coordinates of the data points.
    x (float): The x-value at which to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated value at x.
    """
    assert len(x_points) == len(y_points), "x_points and y_points must have the same length"
    
    n = len(x_points)
    interpolated_value = 0.0
    
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        interpolated_value += term
    
    return interpolated_value

# Example usage
x_points = [1, 2, 3, 4]
y_points = [1, 4, 9, 16]
x = 2.5

interpolated_value = lagrange_interpolation(x_points, y_points, x)
print(f"Interpolated value at x = {x}: {interpolated_value}")

# Plotting
x_values = np.linspace(min(x_points), max(x_points), 100)
y_values = [lagrange_interpolation(x_points, y_points, xi) for xi in x_values]

plt.plot(x_values, y_values, label='Lagrange Interpolation')
plt.scatter(x_points, y_points, color='red', label='Data Points')
plt.scatter([x], [interpolated_value], color='blue', label='Interpolated Value')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Lagrange Interpolation')
plt.show()