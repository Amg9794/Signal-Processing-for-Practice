import numpy as np
import matplotlib.pyplot as plt
from pyfftw.interfaces import scipy_fftpack
import cv2 as cv


def dct_1d(x):
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


def dct_2d(x):
    M, N = x.shape
    X = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        X[m,:] = dct_1d(x[m,:])
    for n in range(N):
        X[:,n] = dct_1d(X[:,n])
    return X

M = 48
N = 32
k1 = 10
k2 = 8

x = np.zeros((M, N), dtype=np.float64)
for m in range(M):
    for n in range(N):
        x[m, n] = np.cos(2*np.pi*k1*m / M)*np.cos(2*np.pi*k2*n / N)

X = dct_2d(x)

print(X)
print(X.shape)

p1, p2 = np.unravel_index(X.argmax(), X.shape)
print("p1: ", p1)
print("p2: ", p2)


## part 2:- DCT from inbuilt function

def dct2(x):
    return scipy_fftpack.dct(scipy_fftpack.dct(x, axis=0, norm='ortho').T, axis=0, norm='ortho').T

X_inbuilt = dct2(x)
print(X_inbuilt)
print(X_inbuilt.shape)


plt.subplot(1,2,1)
plt.imshow(X, cmap='gray')
plt.title("2D_DCT Implemented")
plt.subplot(1,2,2)
plt.imshow((X_inbuilt), cmap='gray')
plt.title("2D DCT inbuilt ")
plt.show()

p1, p2 = np.unravel_index(X_inbuilt.argmax(), X_inbuilt.shape)
print("p1: ", p1)
print("p2: ", p2)


## part 3:- DCT of image Lena 


img = cv.imread('lenna.png', cv.IMREAD_GRAYSCALE)
M, N = img.shape
print(M)
print(N)

dct = dct_2d(img)


plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(np.log(np.abs(dct)), cmap='gray')
plt.title("2D DCT (log scale)")
plt.show()



