# Modules normally used
import numpy as np
import cv2

def CreateFrame(img, krad, color):          # Frame (trick to avoid out-of-bounds access)
    height, width, depth = img.shape

    if color == "white":
        frm = np.ones((height + krad * 2, width + krad * 2, depth))
    else:
        frm = np.zeros((height + krad * 2, width + krad * 2, depth))

    frm[krad:-krad, krad:-krad] = img

    return frm

def FilterImage(img, framed, ksize, krn):   # Apply filter to image
    shape = img.shape
    height = shape[0]
    width = shape[1]

    # Method 1 (Optimal)
    fil = np.zeros(img.shape)
    for i in range(0, height):
        for j in range(0, width):
            fil[i, j] = (framed[i:i+ksize, j:j+ksize] * krn[:, :, np.newaxis]).sum(axis=(0, 1))

    # Method 2 (Visual)
    #for i in range (0, height):
    #    for j in range (0, width):
    #        sub = framed[i:i+ksize, j:j+ksize]
    #        b = (sub[:, :, 0] * krn).sum()
    #        g = (sub[:, :, 1] * krn).sum()
    #        r = (sub[:, :, 2] * krn).sum()
    #        fil[i, j] = (b, g, r)

    return fil

# Normalized Box Filter
def BoxFilter(img, blur):

    # Kernel definition
    ksize = blur                    # kernel size ("diameter")
    krad = int(ksize / 2)           # kernel radius
    krn = np.ones((ksize, ksize))   # create grid
    krn /= krn.sum()                # normalize kernel

    # Frame (trick to avoid out-of-bounds access)
    framed = CreateFrame(img, krad, "black")

    return FilterImage(img, framed, ksize, krn)

def GaussianKernel(ksize):

    krad = int(ksize / 2)
    
    # Method 1 (Optimal)
    x, y = np.meshgrid(np.linspace(-krad, krad, ksize), np.linspace(-krad, krad, ksize))
    d = np.sqrt(x * x + y * y)
    mu, sigma = 0.0, krad / 3   # sigma = 0.4
    krn = np.exp(-((mu - d)**2 / (2 * sigma**2)))
    krn /= krn.sum()
    
    # Method 2 (Visual)
    #krn = np.zeros((ksize, ksize))
    #sigma = krad / 3

    #for i in range (0, ksize):
    #    for j in range (0, ksize):
    #        d = np.sqrt((krad - i)**2 + (krad - j)**2)
    #        krn[i, j] = np.exp(-(d**2 / (2.0 * sigma**2)))

    # Normalize kernel
    krn /= krn.sum()

    return krn

# Normalized Gaussian Filter
def GaussianFilter(img, ksize):

    #Gaussian Kernel
    krad = int(ksize / 2)
    krn = GaussianKernel(ksize)

    # Frame (trick to avoid out-of-bounds access)
    framed = CreateFrame(img, krad, "white")
    
    #Filtered image (output)
    return FilterImage(img, framed, ksize, krn)

def main():
    img = cv2.imread("sonic.jpg", cv2.IMREAD_COLOR)
    img = img / 255.0
    #filtered = BoxFilter(img, 5)
    filtered = GaussianFilter(img, 31)
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(0)

main()