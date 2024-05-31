import numpy as np
import matplotlib.pyplot as plt 

m = 0.1 * 10 ** (-6)
b = 0.2 * 10 ** (-6)
t0 = 0.0 
tf = 10.0
h = 0.01

def noise(m, b, t0, tf, h):
    time_matrix = h * np.abs(np.arange((tf - t0) / h)[:, None] - np.arange((tf - t0) / h)[None, :])
    covariance_matrix = np.zeros_like(time_matrix)

    for i in range(len(time_matrix)):
        for j in range(len(time_matrix)):
            covariance_matrix[i][j] = (2 * m) ** 0.5 * b * np.exp(- m * time_matrix[i][j] * 0.5)
            if (i == j):
                covariance_matrix[i][j] = 1

    L = np.linalg.cholesky(covariance_matrix)
    noise = np.dot(L, np.random.normal(loc = 0, scale = np.sqrt(np.sqrt(2 * m * b ** 2)), size = int((tf - t0) / h)))
    return noise

noise_real = noise(m, b, t0, tf, h)
noise_imag = noise(m, b, t0, tf, h)
noise = np.zeros_like(noise_real, dtype = complex)

for i in range(len(noise)):
    noise[i] = complex(noise_real[i], noise_imag[i])
    noise[i] = (np.absolute(noise[i])) ** 2

t_values = np.linspace(t0, tf, int((tf - t0) / h))

plt.plot(t_values, noise)
plt.xlabel('Time')
plt.ylabel('Noise')
plt.title('Noise against Time')
plt.grid(True)
plt.savefig('noise.png') 

# mean = np.mean(noise)
# variance = np.var(noise)
# print("Mean:", mean)
# print("Variance:", variance)

plt.figure(figsize = (10, 8))

plt.subplot(5, 1, 1)
plt.plot(t_values, noise_real)
plt.xlabel('Time')
plt.ylabel('Noise')
plt.title('Real Noise against Time')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t_values, noise_imag)
plt.xlabel('Time')
plt.ylabel('Noise')
plt.title('Imaginary Noise against Time')
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t_values, noise)
plt.xlabel('Time')
plt.ylabel('Noise')
plt.title('Square of the Magnitude of Complex Noise against Time')
plt.grid(True)
plt.autoscale()
plt.savefig('noise.png')
