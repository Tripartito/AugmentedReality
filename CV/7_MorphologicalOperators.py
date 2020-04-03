# Modules normally used
import numpy as np
import cv2

def erosion(img):
    rows, cols = img.shape
    imgpadding = np.zeros((rows + 2, cols + 2))
    imgpadding[1:-1, 1:-1] = img
    newimg = img.copy()

    for x in range(0, rows):
        for y in range(0, cols):
            if img[x, y] != 0:
                for a in range(x-1, x+1):
                    for b in range(y-1, y+1):
                        if imgpadding[a+1, b+1] == 0:
                            newimg[x, y] = 0

    return newimg

def main():
    img = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)

    erosionx1 = erosion(img)
    erosionx2 = erosion(erosion(img))
    erosionx3 = erosion(erosion(erosion(img)))

    cv2.imshow("original", np.uint8(img))
    cv2.imshow("erosionx1", np.uint8(erosionx1))
    cv2.imshow("erosionx2", np.uint8(erosionx2))
    cv2.imshow("erosionx3", np.uint8(erosionx3))
    cv2.waitKey(0)

main()