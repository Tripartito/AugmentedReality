# Modules normally used
import numpy as np
import cv2

def CreateFrame(img, krad, color):          # Frame (trick to avoid out-of-bounds access)
    height, width = img.shape

    if color == "white":
        frm = np.ones((height + krad * 2, width + krad * 2)) #1 3rd dim for grayscale
    else:
        frm = np.zeros((height + krad * 2, width + krad * 2)) #1 3rd dim for grayscale

    frm[krad:-krad, krad:-krad] = img

    return frm

def MorphologicallyOperate(img, ksize, compare_val):
    krad = int(ksize / 2)           # kernel radius
    framed = CreateFrame(img, krad, "black")

    height, width = img.shape
    fil = img.copy()

    for i in range(0, height):  # Columns loop
        for j in range(0, width):   # Rows loop
            if img[i, j] != compare_val:
                for x in range(i-krad, i+krad):
                    for y in range(j-krad, j+krad):
                        if framed[x+krad, y+krad] == compare_val:
                            fil[i, j] = compare_val

    return fil

def Erosion(img, ksize):
    return MorphologicallyOperate(img, ksize, 0)

def Dilatate(img, ksize):
    return MorphologicallyOperate(img, ksize, 255)

def main():
    img = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)

    erosionx1 = Erosion(img, 2)
    erosionx2 = Erosion(Erosion(img, 2), 2)
    erosionx3 = Erosion(Erosion(Erosion(img, 2), 2), 2)

    cv2.imshow("original", np.uint8(img))
    cv2.imshow("erosionx1", np.uint8(erosionx1))
    cv2.imshow("erosionx2", np.uint8(erosionx2))
    cv2.imshow("erosionx3", np.uint8(erosionx3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dilatex1 = Dilatate(img, 2)
    dilatex2 = Dilatate(Dilatate(img, 2), 2)
    dilatex3 = Dilatate(Dilatate(Dilatate(img, 2), 2), 2)

    cv2.imshow("original", np.uint8(img))
    cv2.imshow("dilatex1", np.uint8(dilatex1))
    cv2.imshow("dilatex2", np.uint8(dilatex2))
    cv2.imshow("dilatex3", np.uint8(dilatex3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()