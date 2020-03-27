# Modules normally used
import numpy as np
import cv2

# Ex 1: Open an image mantaining its original color format and print its internal data type, its
# shape, its number of dimensions, and show it in a window
def ex1():
    img = cv2.imread('sonic.jpg', cv2.IMREAD_ANYCOLOR)  # ANY = Any channels
    dimensions = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (0, 0, 0)
    lineType = 2
    cv2.putText(img, "Type: {}".format(type(img.flatten()[0])), (5, 15), font, fontScale, fontColor, lineType)
    cv2.putText(img, "Shape: {}x{}x{}".format(dimensions[0], dimensions[1], dimensions[2]), (5, 30), font, fontScale, fontColor, lineType)
    cv2.putText(img, "Dimensions: {}".format(len(dimensions)), (5, 45), font, fontScale, fontColor, lineType)
    cv2.imshow('Ex1', img)
    cv2.waitKey(0)

# Ex 2: Open an image mantaining its original clor format, cast it to contain floating point
# values, convert all pixels to range [0, 1], and show the image in a window
def ex2():
    img = cv2.imread('sonic.jpg', cv2.IMREAD_ANYCOLOR)  # ANY = Any channels
    img = np.float64(img)
    img /= img.max()
    cv2.imshow('Ex2', img)
    cv2.waitKey(0)

# Ex 3: Create a binary image (0 and 1s) from an image file and show it
def ex3():
    img = cv2.imread('sonic.jpg', cv2.IMREAD_GRAYSCALE)
    img = np.float64(img)
    img /= img.max()
    img[img >= 0.5] = 1.
    img[img < 0.5] = 0.
    cv2.imshow('Ex3', img)
    cv2.waitKey(0)

# Ex 4: Open an image and apply a vignetting effect on its borders
def ex4():
    img = cv2.imread('sonic.jpg', cv2.IMREAD_ANYCOLOR)
    rows, cols = img.shape[:2]

    # generating vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)

    # applying the mask to each channel in the input image
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask

    cv2.imshow('Original', img)
    cv2.imshow('Vignette', output)
    cv2.waitKey(0)

ex4()