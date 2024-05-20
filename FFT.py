import numpy as np
import matplotlib.pyplot as plt

# Parameters
sampling_rate = 1000  # Hz
T = 1.0 / sampling_rate  # Sampling interval
t = np.arange(0, 1.0, T)  # Time vector

# Create a signal composed of multiple sine waves
freq1 = 50  # Frequency of the first sine wave (Hz)
freq2 = 120  # Frequency of the second sine wave (Hz)
amplitude1 = 1.0  # Amplitude of the first sine wave
amplitude2 = 0.5  # Amplitude of the second sine wave
signal = amplitude1 * np.sin(2 * np.pi * freq1 * t) + amplitude2 * np.sin(2 * np.pi * freq2 * t)

# Apply FFT
n = len(t)  # Number of samples
fhat = np.fft.fft(signal, n)  # Compute the FFT
psd = fhat * np.conj(fhat) / n  # Power spectral density
freq = (1 / (T * n)) * np.arange(n)  # Frequency array
L = np.arange(1, np.floor(n / 2), dtype='int')  # Only use the first half of the FFT output

# Plot the signal
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain Signal')
plt.legend()
plt.grid(True)

# Plot the frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(freq[L], psd[L], label='Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Frequency Domain Signal')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
