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

def MedianFilter(img, size):
    radius = int(size / 2)           # Effect Radius

    # Frame (trick to avoid out-of-bounds access)
    framed = CreateFrame(img, radius, "black")

    shape = img.shape
    height = shape[0]
    width = shape[1]

    fil = np.zeros(img.shape)

    #Apply Filter
    for i in range (0, height):
        for j in range (0, width):
            sub = framed[i:i+size, j:j+size]
            b = np.median(sub[:, :, 0])
            g = np.median(sub[:, :, 1])
            r = np.median(sub[:, :, 2])
            fil[i, j] = (b, g, r)

    return fil

def BilateralFilter(img, tex, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))

    # Image padding
    if img.ndim == 3:
        img_height = img.shape[0]
        img_width = img.shape[1]
        I = np.pad(img, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif img.ndim == 2:
        img_height = img.shape[0]
        img_width = img.shape[1]
        I = np.pad(img, ((r, r), (r, r)), 'symmetric').astype(np.float32)

    # Check texture size and do padding
    if tex.ndim == 3:
        tex_height = tex.shape[0]
        tex_width = tex.shape[1]
        if tex_height != img_height or tex_width != img_width:
            print('The guidance image is not aligned with input image!')
            return img
        T = np.pad(tex, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif tex.ndim == 2:
        tex_height = tex.shape[0]
        tex_width = tex.shape[1]
        if tex_height != img_height or tex_width != img_width:
            print('The guidance image is not aligned with input image!')
            return img
        T = np.pad(tex, ((r, r), (r, r)), 'symmetric').astype(np.int32)

    # Pre-compute
    output = np.zeros_like(img)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)

    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)

    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    
    # Main body
    if I.ndim == 2 and T.ndim == 2:     # I1T1 filter
        for y in range(r, r + img_height):
            for x in range(r, r + img_width):
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif I.ndim == 3 and T.ndim == 2:     # I3T1 filter
        for y in range(r, r + img_height):
            for x in range(r, r + img_width):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 3 and T.ndim == 3:     # I3T3 filter
        for y in range(r, r + img_height):
            for x in range(r, r + img_width):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 2 and T.ndim == 3:     # I1T3 filter
        for y in range(r, r + img_height):
            for x in range(r, r + img_width):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return img

    # return np.clip(output, 0, 255)
    return output

def main():
    img = cv2.imread("valley.jpg", cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    #filtered = MedianFilter(img, 5)
    filtered = BilateralFilter(img, img_gray, 1, 0.1*255)
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", filtered)
    cv2.waitKey(0)

main()

### Ref Code ###
# # image: input image
# # texture: guidance image
# # sigma_s: spatial parameter (pixels)
# # sigma_r: range parameter (not normalized)
# def bilateralfilter(image, texture, sigma_s, sigma_r):
#     r = int(np.ceil(3 * sigma_s))
#     # Image padding
#     if image.ndim == 3:
#         h, w, ch = image.shape
#         I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
#     elif image.ndim == 2:
#         h, w = image.shape
#         I = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
#     else:
#         print('Input image is not valid!')
#         return image
#     # Check texture size and do padding
#     if texture.ndim == 3:
#         ht, wt, cht = texture.shape
#         if ht != h or wt != w:
#             print('The guidance image is not aligned with input image!')
#             return image
#         T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
#     elif texture.ndim == 2:
#         ht, wt = texture.shape
#         if ht != h or wt != w:
#             print('The guidance image is not aligned with input image!')
#             return image
#         T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
#     # Pre-compute
#     output = np.zeros_like(image)
#     scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
#     scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
#     # A lookup table for range kernel
#     LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
#     # Generate a spatial Gaussian function
#     x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
#     kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
#     # Main body
#     if I.ndim == 2 and T.ndim == 2:     # I1T1 filter
#         for y in range(r, r + h):
#             for x in range(r, r + w):
#                 wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
#                 output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
#     elif I.ndim == 3 and T.ndim == 2:     # I3T1 filter
#         for y in range(r, r + h):
#             for x in range(r, r + w):
#                 wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
#                 wacc = np.sum(wgt)
#                 output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
#                 output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
#                 output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
#     elif I.ndim == 3 and T.ndim == 3:     # I3T3 filter
#         for y in range(r, r + h):
#             for x in range(r, r + w):
#                 wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
#                       LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
#                       LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
#                       kernel_s
#                 wacc = np.sum(wgt)
#                 output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
#                 output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
#                 output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
#     elif I.ndim == 2 and T.ndim == 3:     # I1T3 filter
#         for y in range(r, r + h):
#             for x in range(r, r + w):
#                 wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
#                       LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
#                       LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
#                       kernel_s
#                 output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
#     else:
#         print('Something wrong!')
#         return image

#     # return np.clip(output, 0, 255)
#     return output


# if __name__ == '__main__':
#     sigma_s = 1
#     sigma_r = 0.1*255
#     img = cv2.imread('2c.png')
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     tic = time.time()
#     img_bf = bilateralfilter(img, img_gray, sigma_s, sigma_r)
#     toc = time.time()
#     print('Elapsed time: %f sec.' % (toc - tic))
#     cv2.imwrite('2c_y.png', img_gray)
#     cv2.imwrite('output.png', img_bf)