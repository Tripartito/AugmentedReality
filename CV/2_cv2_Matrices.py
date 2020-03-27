# Modules normally used
#import numpy as np
import cv2

# Load an image in grayscale
img = cv2.imread('img_name.jpg', 0)

print(type(img))
print(img.shape[0])
print(img.shape[1])
print(img.shape[2])
height, width, depth = img.shape

k = cv2.waitKey(0)

if k == 27:
    # Wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    # Wait for 's' key to save and exit
    cv2.imwrite('grayscale_version.jpg', img)
    cv2.destroyAllWindows()