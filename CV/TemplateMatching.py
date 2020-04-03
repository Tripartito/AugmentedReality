# Template Matching Algorithm
import cv2
import numpy as np

def TemplateMatching(target,img):
    
    # Print image and target dimensions
    print('Image Height       : ',img.shape[0])
    print('Image Width        : ',img.shape[1])

    print('target Height       : ',target.shape[0])
    print('target Width        : ',target.shape[1])

    # Template match dimensions
    width = img.shape[1] - target.shape[1] + 1
    height = img.shape[0] - target.shape[0] + 1

    # Build Template match
    filter = np.zeros((height, width))

    for i in range (0, height):
        for j in range (0, width):
            filter[i,j] = np.sum(target[:] - img[i:i + target.shape[1], j: j + target.shape[0]])**2

    return filter

def downscale(img, scalefactor):
    width = int(img.shape[1] * scalefactor / 100)
    height = int(img.shape[0] * scalefactor / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)

    return img

def upscale(img, scalefactor):
    width = int(img.shape[1] / (scalefactor / 100))
    height = int(img.shape[0] / (scalefactor / 100))
    dim = (width, height)
    img = cv2.resize(img, dim)

    return img

# Define threshold and allow input
threshold = 0.1
threshhold = input("Input threshold")

# Define scale factor % (to improve performance) 
# CAREFUL, if factor is too low you may find undesired behaviour!! (erroneous results)
scalefactor = 50
scalefactor = input("Provide % to downscale images for performance (may yield worse results), type 100 to prevent downscaling: ")

scalefactor = int(scalefactor)

# Ask user to provide path to image and target
imgpath = input("Provide image path (relative works, include extension)")
targetpath = input("Provide target path (relative works, include extension)")

print("Threshold chosen: ", threshhold)
print("Image chosen: ", imgpath)
print("Target chosen", targetpath)

# Read original image and target as grayscales 
img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
target = cv2.imread(targetpath, cv2.IMREAD_GRAYSCALE)

# Read original image in rgb to print later 
imgcolor = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR)

# Scale down images so we process them faster
img = downscale(img, scalefactor)
target = downscale(target, scalefactor)
imgcolor = downscale(imgcolor, scalefactor)

# Obtain template match and normalize it
filtered = TemplateMatching(target,img)
filtered = filtered / filtered.max()

# Create a black image
imgFound = np.zeros((40,245,3), np.uint8)

# Write the sentence in the img
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imgFound, "TARGET_FOUND", (5,30), font, 1, (0,255,0), 2)

# Display the TARGET_FOUND image if target is found
if (np.min(filtered)/np.max(filtered)) < threshold:
    cv2.imshow("Result", imgFound)
    print("TARGET FOUND")

# Retrieve where in the matching map there are values below threshold
loc = np.where( filtered < threshold)


# Draw rectangles in these places (on the coloured original image!)
for point in zip(*loc[::-1]): # Notice we retrieve a point from the list at each iteration, and use it to draw a rectangle
    cv2.rectangle(imgcolor, point, (point[0] + target.shape[0], point[1] + target.shape[1]), (0,255,0), 2)

# Scale up images for display
img = upscale(img, scalefactor)
target = upscale(target, scalefactor)
imgcolor = upscale(imgcolor, scalefactor)

# Show images
cv2.imshow("Target",target)
cv2.imshow("Original",imgcolor)
cv2.imshow("Matching map", filtered)
cv2.waitKey(0)

