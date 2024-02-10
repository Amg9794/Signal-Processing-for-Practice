import numpy as np
import matplotlib.pyplot as plt


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

# zero pad both sequences
x = np.pad(x,(0,N-len(x)),'constant')
h = np.pad(h,(0,N-len(h)),'constant')

print(len(x))
print(len(h))


def dft(x):
    # Initialize an array to store the DFT coefficients
    X = np.zeros(len(x), dtype=np.complex)
    N = len(x)
    for k in range(N):
        for n in range(N):
            X[k] += x[n]*np.exp(-1j*2*np.pi*k*n / N)
    return X

def idft(X):
    # Initialize an array to store the IDFT coefficients
    y = np.zeros(len(X), dtype=np.complex)
    N = len(X)
    for n in range(N):
        for k in range(N):
            y[n] += X[k]*np.exp(1j*2*np.pi*k*n / N)
    return y / N

# Find the DFT of x[n]
X = dft(x)
H = dft(h)

# Find the IDFT of X
y = idft(X)

# Multiply the DFTs element-wise
Y = X*H

# Take the IDFT of the product
y = idft(Y)
print(y)
print(len(y))

plt.title('Magnitude plot of X[k]')
plt.plot(t,np.abs(X))
plt.xlabel('Frequency (k)')
plt.ylabel('Magnitude')
plt.show()

plt.title('Magnitude plot of H[k]')
plt.plot(t,np.abs(H))
plt.xlabel('Frequency (k)')
plt.ylabel('Magnitude')
plt.show()

plt.title('Magnitude plot of Y[k]')
plt.plot(t,np.abs(Y))
plt.xlabel('Frequency (k)')
plt.ylabel('Magnitude')
plt.show()