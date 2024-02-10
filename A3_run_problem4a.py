import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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


def inverse_2d_dct(dct_matrix):
    M, N = dct_matrix.shape
    idct_matrix = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            sum = 0
            for k1 in range(M):
                c1 = np.sqrt(2/M) if k1 != 0 else 1/np.sqrt(M)
                for k2 in range(N):
                    c2 = np.sqrt(2/N) if k2 != 0 else 1/np.sqrt(N)
                    sum += dct_matrix[k1,k2]*c1*np.cos((np.pi*k1*(2*m+1))/M)*c2*np.cos((np.pi*k2*(2*n+1))/N)
            idct_matrix[m][n] = sum
    return idct_matrix


# Cosine Image
k1 = 10
k2 = 8
M = 48
N = 32
cos_img = np.zeros((M, N))
for m in range(M):
    for n in range(N):
        cos_img[m,n] = np.cos(2*np.pi*k1*m/M) * np.cos(2*np.pi*k2*n/N)

#Compute DCT
dct_cos_img = dct_2d(cos_img)
plt.imshow(dct_cos_img, cmap='gray')
plt.show()

# Compute Inverse DCT
idct_cos_img = inverse_2d_dct(dct_cos_img)

# Plot Original and Inverted Images
plt.subplot(121)
plt.imshow(cos_img, cmap='gray')
plt.title('Original cosine Image')

plt.subplot(122)
plt.imshow(idct_cos_img, cmap='gray')
plt.title('Inverted cosine Image')
plt.show()

# Lena Image
img1 = cv.imread('Grayscale-Lena-Image.png',cv.IMREAD_GRAYSCALE)

# Compute DCT
dct_lena = dct_2d(img1)

# Compute Inverse DCT
idct_lena = inverse_2d_dct(dct_lena)

# Plot Original and Inverted Images
plt.subplot(121)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(idct_lena, cmap='gray')
plt.title('Inverted Image')
plt.show()





