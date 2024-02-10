
from pyfftw.interfaces import scipy_fftpack
import numpy as np
import matplotlib.pyplot as plt

N = 32
n= np.linspace(0,1000,N)
k = 5
xs = np.cos(2*np.pi*k*n/N)
print(xs)
print(len(xs))

plt.plot(n,xs)
plt.title('Orginal Function')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.show()


def dct(x):
    N = len(x)
    X = np.zeros(N, dtype=np.float64)
    for k in range(N):
        sum = 0.0
        for n in range(N):
            sum += x[n]*np.cos((np.pi*k*(2*n + 1)) / (2 * N))
        if k != 0:
            X[k] = sum * np.sqrt(2.0 / N)
        else:
            X[k] = sum * np.sqrt(1/N)
    return X



dct_xs = dct(xs)
print(dct_xs)

##   using inbuilt function
def dct2(x):
    return scipy_fftpack.dct(x, norm='ortho')


dct_xs_inbuilt = dct2(xs)
print(dct_xs_inbuilt)

plt.subplot(1,2,1)
plt.stem(dct_xs)
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Discrete Cosine Transform')
# plt.show()
plt.subplot(1,2,2)
plt.stem(dct_xs_inbuilt)
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Discrete Cosine Transform using inbuilt function')
plt.show()

