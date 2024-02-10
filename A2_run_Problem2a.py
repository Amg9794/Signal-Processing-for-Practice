import numpy as np


fs = 200  
f = 20  
T = 1/fs  
t = np.arange(0, 1, T)  
m = 0  # Mean of noise
var = 0.04  # Variance of noise

# Generate signal x[n]
sin_component = np.sin(2*np.pi*f*t)
noise_component = np.random.normal(m, np.sqrt(var), t.shape)
x = sin_component + 0.8*noise_component

# Define impulse response h[n]
h = np.array([1/3, 1/3, 1/3])


# Find the length of the sequences
N = len(x)
M = len(h)

h = np.pad(h, (0, N-M), 'constant')

# Create the circulant matrix of h[n]
Cir_Mat = np.zeros((N, N))
for i in range(N):
    Cir_Mat[i, :] = np.roll(h, i)

# Create the column vector x
x = np.resize(x, (N, 1))

# Multiply the circulant matrix and the column vector to get the convolution
y = np.matmul(Cir_Mat, x)

print(y)
print(len(y))