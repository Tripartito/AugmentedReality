# Modules normally used
import numpy as np
import cv2

# Ex 1: Create a vector of 10 floats (zeros)
def ex1():
    arr = np.zeros(shape=(1, 10))  # If you know the param name you can speciy it on the func call
    print(arr)

# Ex 2: Create a vector of 10 float (zeroes) whose 5th element is one
def ex2():
    arr = np.zeros(shape=(1, 10))  # If you know the param name you can speciy it on the func call
    arr[0, 4] = 1.0
    print(arr)

# Ex 3: Create a vector of integers going from 10 to 49
def ex3():
    arr = np.arange(10, 50)
    print(arr)

# Ex 4: Create a matrix 3x3 floats going from 1 to 9
def ex4():
    arr = np.arange(1, 10)
    arr = arr.reshape((3, 3))
    print(arr)

# Ex 5: Create a matrix of 3x3 floats going from 1 to 9 and flip it horizontally
def ex5():
    arr = np.arange(1, 10)
    arr = arr.reshape((3, 3))
    arr = np.flip(arr, axis=1)  # If you know the param name you can speciy it on the func call
    print(arr)

# Ex 6: Create a matrix of 3x3 floats going from 1 to 9 and flip it vertically
def ex6():
    arr = np.arange(1, 10)
    arr = arr.reshape((3, 3))
    arr = np.flip(arr, axis=0)  # If you know the param name you can speciy it on the func call
    print(arr)

# Ex 7: Create a 3x3 identity matrix
def ex7():
    arr = np.identity(3)
    print(arr)

# Ex 8: Create a 3x3 of random values
def ex8():
    arr = np.random.rand(3, 3)
    print(arr)

# Ex 9: Create a random vector of 10 numbers and compute the mean value
def ex9():
    arr = np.random.randint(0, 101, 10)
    print(arr)
    print(arr.mean())

# Ex 10: Create a 10x10 array of zeros surrounded/frames by ones
def ex10():
    arr = np.ones(shape=(10, 10))
    arr[1:-1, 1:-1] = 0
    print(arr)

# Ex 11: Create a 5x5 matrix of rows from 1 to 5
def ex11():
    arr = np.ones((5, 5))
    arr[0:5, :] = [1, 2, 3, 4, 5]
    print(arr)
    # OR BETTER
    arr = np.ones((5, 5))
    arr += np.arange(5)
    print(arr)

# Ex 12: Create an array of 9 random integers and reshape it to a 3x3 matrix of floats
def ex12():
    arr = np.random.randint(0, 10, 9)
    print(arr)
    arr = np.float64(arr)
    #arr = arr.reshape((3, 3)) # Commented for error prompt, but works
    print(arr)

# Ex 13: Create a 5x5 matrix of random values and substract its average from it
def ex13():
    arr = np.random.rand(5, 5)
    print(arr.mean())

# Ex 14: Create a 5x5 matrix of random values and substract the average of each row to each row
def ex14():
    arr = np.random.rand(5, 5)
    print(arr)
    arr[0:5, :] -= arr[1, :].mean()
    print(arr)

# Ex 15: Create an array of 5x5 random values and return the value that is close to 0.5
def ex15():
    arr = np.random.rand(5, 5)
    idx = (np.abs(arr - 0.5)).argmin()  # Create an array that stores each number's distance to 0.5 and extract the min value
    print(arr)
    print(arr.flatten()[idx])

# Ex 16: Make a 3x3 matrix of random numbers from 0 to 10 and count how many of them are > 5
def ex16():
    arr = np.random.randint(0, 10, (3, 3))
    print(arr)
    print((arr[arr > 5].size))

# Ex 17: Create a horizontal gradient image of 64x64 that goes from black to white
def ex17():
    img = np.tile(np.linspace(0, 255, 64), (64, 1)) # Method 1
    cv2.imshow('Ex17', np.uint8(img))
    cv2.waitKey(0)

# Ex 18: Create a vertical gradient image of 64x64 that goes from black to white
def ex18():
    img = np.zeros((64, 64))                        # Method 2
    grad = np.arange(0., 1., 1./64.)
    grad = grad.reshape((64, 1))    # From horizontal to vertical
    img += grad
    cv2.imshow('Ex18', img)
    cv2.waitKey(0)

# Ex 19: Create a 3-component white image of 64x64 pixels, set the blue component to zero
def ex19():
    size = 64
    image = np.empty((size, size, 3), np.uint8)
    image[:] = 255
    image[:, :, 0] = 0     # (B, G, R)
    cv2.imshow('Ex19', image)
    cv2.waitKey(0)

# Ex 20: Create a 3-component white image of 64x64 pixels,
# set the blue component of the top-left part to 0 (yellow) 
# and the red component of the bottom-right part to zero (cyan)
def ex20():
    size = 64
    image = np.empty((size, size, 3), np.uint8)
    image[:] = 255
    image[0:31, 0:31, 0] = 0
    image[32:63, 32:63, 2] = 0
    cv2.imshow('Ex20', image)
    cv2.waitKey(0)

# Ex 21: Open an image and insert black horizontal scan lines at 50%
def ex21():
    image = cv2.imread('sonic.jpg', cv2.IMREAD_COLOR)
    image[::2, :] = 0.0
    cv2.imshow('Ex21', image)
    cv2.waitKey(0)
    
# Ex 22: Open an image and insert black vertical scan lines at 50%
def ex22():
    image = cv2.imread('sonic.jpg', cv2.IMREAD_COLOR)
    image[:, ::2] = 0.0
    cv2.imshow('Ex22', image)
    cv2.waitKey(0)

# Ex 23: Open an image, convert it to float64, normalize it, darken it 50%, and show it
def ex23():
    image = cv2.imread('sonic.jpg', cv2.IMREAD_COLOR)   # NO ANY = always 3 channels
    image = np.float64(image)
    image /= image.max()
    image *= 0.5
    cv2.imshow('Ex23', image)
    cv2.waitKey(0)