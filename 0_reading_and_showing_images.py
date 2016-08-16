# imports
import numpy as np
import cv2

# load a grayscale image
img = cv2.imread('images/dolphin.png')

# display the image with imshow
# first argument is the window name
# second argument is the image to show
# you can create as many images as you want, they just have to have different
# window names
cv2.imshow('image', img)

# a keyboard binding function
# Its argument is the time in milliseconds. The function waits for specified
# milliseconds for any keyboard event. If you press any key in that time,
# the program continues. If 0 is passed, it waits indefinitely for a key stroke.
# It can also be set to detect specific key strokes like, if key a is pressed etc
# which we will discuss below.
cv2.waitKey(0)

# *Note*: Besides binding keyboard events this function also processes many
# other GUI events, so you MUST use it to actually display the image.

# destroy all windows command
# simply destroys all the windows we created. If you want to destroy any specific
# window, use the function cv2.destroyWindow() where you pass the exact window
# name as the argument.
cv2.destroyAllWindows()

# *Note*: There is a special case where you can already create a window and load
# image to it later. In that case, you can specify whether window is resizable
# or not. It is done with the function cv2.namedWindow(). By default, the flag
# is cv2.WINDOW_AUTOSIZE. But if you specify flag to be cv2.WINDOW_NORMAL,
# you can resize window. It will be helpful when image is too large in dimension
# and adding track bar to windows.

cv2.namedWindow('image 2 named Window', cv2.WINDOW_NORMAL)
cv2.imshow('image 2 Show',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write an image with imwrite
# first argument is the file name
# second argument is the image you want to save
cv2.imwrite('images/dolphin_2.png', img)

# summing it up all together
img = cv2.imread('images/dolphin.png')
cv2.imshow('Option to Save image', img)
key = cv2.waitKey(0) # f you are using a 64-bit machine key = cv2.waitKey(0) & 0xFF
if key == 27: # wait for the ESC key to exit
    cv2.destroyAllWindows()
elif key == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('images/dolphin_3.png', img)
    cv2.destroyAllWindows()


# using matplotlib to display the image
# *WARNING*: Color image loaded by OpenCV is in BGR mode. But Matplotlib displays
# in RGB mode. So color images will not be displayed correctly in Matplotlib if
# image is read with OpenCV. Please see the exercises for more details.
import matplotlib.pyplot as plt

img = cv2.imread('images/dolphin.png')
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # hide x and y tick values
plt.show()

# get the image size
height, width = img.shape[:2]
print 'Image Width: {}px, Image Height: {}px'.format(width, height)
print type(img) # openCV stores images as np.ndarray
