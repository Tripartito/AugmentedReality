# Modules normally used
import numpy as np
import cv2

def applyFilter(img, filter):
    newimg = np.float64(img.copy())
    rows, columns = img.shape
    f_rows, f_columns = filter.shape
    f_rows_half = np.uint8(f_rows / 2)
    f_columns_half = np.uint8(f_columns / 2)
    for x in range(0, rows):
        for y in range(0, columns):
            submat = img[max(0, x-f_rows_half):min(rows, x+f_rows_half+1), max(0, y-f_columns_half):min(columns, y+f_columns_half+1)]
            f_submat = filter[max(f_rows_half-x, 0):f_rows-max(0, x+f_rows_half-rows+1), max(f_columns_half-y, 0):f_columns-max(0, y+f_columns_half-columns+1)]
            newimg[x, y] = np.sum(submat*f_submat)
    return newimg

def applyGaussian(img):

    # Create the kernel manually
    kernel = np.array((
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ), float)
    kernel /= 159
    return applyFilter(img, kernel)

def sobel_gx(img):
    kernelHorizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return applyFilter(img, kernelHorizontal)

def sobel_gy(img):
    kernelVertical = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    return applyFilter(img, kernelVertical)

def sobel_G(gx, gy):
    return np.sqrt(gx**2+gy**2)

def getAngle(gx, gy):   #calculates the direction from gx and gy and rounds the angle to 0,45,80,135
    angle = np.arctan2(gx, gy)
    rows, cols = angle.shape
    for x in range(0, rows):
        for y in range(0, cols):
            if angle[x, y] < 0:
                angle[x, y] += np.pi

            if abs(angle[x, y]) < np.pi/8: #between 0 and 22,5
                angle[x, y] = 0

            elif abs(angle[x, y]) - np.pi/4 < np.pi/8: #between 22,5 and 67,5
                angle[x, y] = np.pi/4

            elif abs(angle[x, y]) - np.pi/2 < np.pi/8: #between 67,5 and 112,5
                angle[x, y] = np.pi/2

            elif abs(angle[x, y]) - np.pi*3/4 < np.pi/8: #between 112,5 and 157,5
                angle[x, y] = np.pi*3/4

            elif abs(angle[x, y]) - np.pi < np.pi /8: #between 157,5 and 180
                angle[x, y] = 0
    return angle

def thinEdges(g, angle):
    rows, cols = g.shape
    imgpadding = np.zeros((rows+2, cols+2)) #add borders
    imgpadding[1:-1, 1:-1] = g #g with borders

    matrix = np.zeros((rows, cols))

    for x in range(0, rows):
        for y in range(0, cols):
            if angle[x, y] == 0:
                if g[x, y] > imgpadding[x, y+1] and g[x, y] > imgpadding[x+2, y+1]:
                    matrix[x, y] = g[x, y]

            if angle[x, y] == np.pi/4:
                if g[x, y] > imgpadding[x, y] and g[x, y] > imgpadding[x+2, y+2]:
                    matrix[x, y] = g[x, y]

            if angle[x, y] == np.pi/2:
                if g[x, y] > imgpadding[x+1, y] and g[x,y] > imgpadding[x+1, y+2]:
                    matrix[x, y] = g[x, y]

            if angle[x, y] == np.pi*3/4:
                if g[x, y] > imgpadding[x+2, y] and g[x, y] > imgpadding[x, y+2]:
                    matrix[x, y] = g[x, y]
    return matrix


def trueEdges(thin, min, max):
    rows, cols = thin.shape
    hysteresis = np.zeros((rows, cols))

    for x in range(0, rows):
        for y in range(0, cols):
            if thin[x, y] > max:
                hysteresis[x, y] = 255

    for x in range(0, rows):
        for y in range(0, cols):
            if thin[x, y] > min and thin[x, y] < max:
                for i in range(-1, 1):
                    for j in range(-1, 1):
                        if hysteresis[x+i, y+j] == 255:
                            hysteresis[x, y] == 255

    return hysteresis

def cannyEdges(img, min, max):

    gx = sobel_gx(img)
    gy = sobel_gy(img)

    G = sobel_G(gx, gy)
    Ang = getAngle(gx, gy)

    Thinedge = thinEdges(G, Ang)
    Truedge = trueEdges(Thinedge, min, max)

    cv2.imshow("original", np.uint8(img))
    cv2.imshow("gx", np.uint8(gx))
    cv2.imshow("gy", np.uint8(gy))
    cv2.imshow("G", np.uint8(G))
    cv2.imshow("thin", np.uint8(Thinedge))
    cv2.imshow("cannyedges", np.uint8(Truedge))
    cv2.waitKey(0)

def main():
    img = cv2.imread("sonic.jpg", cv2.IMREAD_GRAYSCALE)
    cannyEdges(img, 25, 100)
    cv2.waitKey(0)

main()