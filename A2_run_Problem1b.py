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
L = len(x)
M = len(h)

N = L+M-1

# Create the Toeplitz matrix of h[n]
Toep_Mat = np.zeros((N, L))
for i in range(N):
    for j in range(L):
        if i-j>=0 and i-j<M:
            Toep_Mat[i,j] = h[i-j]

# Create the column vector x
x = np.resize(x, (L, 1))

# Multiply the Toeplitz matrix and the column vector to get the convolution
y = np.matmul(Toep_Mat, x)

print(y)
print(len(y))
