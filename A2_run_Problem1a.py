import numpy as np
import matplotlib.pyplot as plt

fs = 200  # Sampling frequency
f = 20  # Frequency of sinusoid
T = 1/fs  # Sampling period
t = np.arange(0, 1, T)  # Array of sample points in time domain
m = 0  # Mean of noise
var = 0.04  # Variance of noise

# Generate signal x[n]
sin_component = np.sin(2*np.pi*f*t)
noise_component = np.random.normal(m, np.sqrt(var), t.shape)
x = sin_component + 0.8*noise_component
print(x)
plt.stem(t,x,'r')
plt.title('Sampled Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


# Define impulse response h[n]
h = np.array([1/3, 1/3, 1/3])

# Compute y[n] = (x * h)[n]
def Brute_Force_Conv(x,h):
    L = (len(x))
    M = (len(h))
    N = int(L+M -1)
    y = np.zeros(N)
    m = int(N-L)
    n = int(N-M)
    x = np.pad(x,(0,m),'constant')
    h = np.pad(h,(0,n),'constant')
    #Linear convolution using convolution sum formula
    for n in range (N):
        for k in range (N):
          if n >= k:
             y[n] = y[n]+x[n-k]*h[k]
    return y      
linear_conv1 = Brute_Force_Conv(x,h)
print('Linear convolution using Brute Force Method output response y =\n',linear_conv1)
# plt.stem(t,linear_conv1,'r')
# plt.title('y[n] = (x * h)[n]')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()

