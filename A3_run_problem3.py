import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# 1D_DCT_Computation
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
            X[k] = sum / np.sqrt(N)
    return X


##  2D_DCT Computation
def compute_2ddct(x):
    M, N = x.shape
    for m in range(M):
        x[m, :] = dct(x[m, :])
    for n in range(N):
        x[:, n] = dct(x[:, n])
    return x


#Extracting 8x8 patches from image
def extract_patches(img, patch_size=(8, 8)):
    M, N = img.shape
    m_patch, n_patch = patch_size
    patches = []
    for i in range(0, M-m_patch+1, m_patch):
        for j in range(0, N-n_patch+1, n_patch):
            patch = img[i:i+m_patch, j:j+n_patch]
            patches.append(patch)
    return np.array(patches)


# stitching patches back to get orginal image
def stitch_patches(patches, img_size):
    M, N = img_size
    patches = np.stack(patches)
    m_patch, n_patch = patches.shape[1:]

    m_patch, n_patch = patches.shape[1:]
    img = np.zeros(img_size)
    k = 0
    for i in range(0, M-m_patch+1, m_patch):
        for j in range(0, N-n_patch+1, n_patch):
            img[i:i+m_patch, j:j+n_patch] = patches[k]
            k += 1
    return img


# calculating 2D_DCT on each patch then stiching them back
img = cv.imread('pepper.png',cv.IMREAD_GRAYSCALE)
patches = extract_patches(img)
dct_patches = [compute_2ddct(patch) for patch in patches]
dct_img = stitch_patches(dct_patches, img.shape)
log_dct_img = np.log(np.abs(dct_img) + 1e-8)  # adding some positive value to avoid logarithm of zero and Neg value



plt.subplot(1,2,1)
plt.imshow(img,cmap ='gray')
plt.title('Orginal Pepper img')
plt.subplot(1,2,2)
plt.imshow(log_dct_img,cmap ='gray')
plt.title('Block DCT of Pepper (log scale)')
plt.show()
