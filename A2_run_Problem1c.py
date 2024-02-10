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
print(len(x))

# Define impulse response h[n]
h = np.array([1/3, 1/3, 1/3])

# Pad x[n] and h[n] with zeros
L = len(x)
M = len(h)
if L > M:
    pad_length = (L - M) // 2
    h = np.pad(h, (pad_length, L - M - pad_length), 'constant')
    
else:
    pad_length = (M - L) // 2
    x = np.pad(x, (pad_length, M - L - pad_length), 'constant')
   
print(len(h))    
print(len(x)) 

# Perform DFT of x[n]
N = len(x)
X = np.zeros(N, dtype=complex)
for k in range(N):
    for n in range(N):
        X[k] = X[k]+x[n]*np.exp(-2j*np.pi*k*n/N)

# Perform DFT of h[n]
M = len(h)
H = np.zeros(M, dtype=complex)
for k in range(M):
    for n in range(M):
        H[k] = H[k]+h[n]*np.exp(-2j*np.pi*k*n/M)


#Element wise multiplication
Y = X*H

# Perform IDFT of Y[k]
N = len(X)
y = np.zeros(N, dtype=complex)
for n in range(N):
    for k in range(N):
        y[n] = y[n]+Y[k]*np.exp(2j*np.pi*k*n/N)
y = y / N
y = np.real(y)
print(y)
print(len(y))

# Calculate magnitude responses
X_magnitude = np.abs(X)
H_magnitude = np.abs(H)
Y_magnitude = np.abs(Y)

plt.title('Magnitude plot of x[k]')
plt.plot(t,X_magnitude)
plt.xlabel('Frequency (k)')
plt.ylabel('Magnitude')
plt.show()

# Plot magnitude responses
# plt.figure()
# plt.plot(X_magnitude, label='X[k]')
# plt.plot(H_magnitude, label='H[k]')
# plt.plot(Y_magnitude, label='Y[k]')
# plt.xlabel('Frequency (k)')
# plt.ylabel('Magnitude')
# plt.legend()
# plt.grid()
# plt.show()
