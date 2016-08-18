###############################################################################
#   BASICS 0.0: Reading, Showing & Saving Images with OpenCV and Matplotlib   #
#                               by: Todd Farr                                 #
###############################################################################

# imports
import numpy as np
import cv2

################################################################# READING IMAGES
print 'Loading Images: \n'

# load an image
# NOTE: OpenCV will import all images (grayscale or color) as having 3 channels,
# to read an image only as a single channel pass the arg 0 after the image location
img = cv2.imread('images/dolphin.png')
img_single_channel = cv2.imread('images/dolphin.png', 0)

print 'The shape of img without second arg is: {}'.format(img.shape)
print 'The shape of img_single_channel is:     {}\n'.format(img_single_channel.shape)


################################################################# DISPLAY IMAGES
print 'Display Images using OpenCV imshow(): \n'

# display the image with OpenCV imshow()
#### 1st ARGUMENT --> the window name
#### 2nd ARGUMENT --> the image to show
# You can show as many images as you want at once, they just have to have
#different window names
cv2.imshow('OpenCV imshow()', img)

# OpenCV waitKey() is a required keyboard binding function after imwshow()
# Its argument is the time in milliseconds. The function waits for specified
# milliseconds for any keyboard event. If you press any key in that time,
# the program continues. If 0 is passed, it waits indefinitely for a key stroke.
# It can also be set to detect specific key strokes like if key a is pressed etc.
cv2.waitKey(0)

# NOTE: Besides binding keyboard events this waitKey() also processes many
# other GUI events, so you MUST use it to actually display the image.

# destroy all windows command
# simply destroys all the windows we created. If you want to destroy any specific
# window, use the function cv2.destroyWindow() where you pass the exact window
# name as the argument.
cv2.destroyAllWindows()

# NOTE: There is a special case where you can already create a window and load
# image to it later. In that case, you can specify whether window is resizable
# or not. It is done with the function cv2.namedWindow(). By default, the flag
# is cv2.WINDOW_AUTOSIZE. But if you specify flag to be cv2.WINDOW_NORMAL,
# you can resize window. It will be helpful when image is too large in dimension
# and adding track bar to windows.

cv2.namedWindow('Named-Empty Resizable Window', cv2.WINDOW_NORMAL)
cv2.imshow('Dolphins are awesome!',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


################################################################## SAVING IMAGES
print 'Saving Images using OpenCV imwrite():'

# write an image with imwrite
# first argument is the file name
# second argument is the image you want to save
cv2.imwrite('images/dolphin_2.png', img)

# summing it up all together, saving on 's'
img = cv2.imread('images/dolphin.png')
cv2.imshow('Option to Save image', img)
print "press 's' to save the image as dolphin_3.png\n"
key = cv2.waitKey(0) # NOTE: if you are using a 64-bit machine see below.
# The above line needs to be: key = cv2.waitKey(0) & 0xFF
if key == 27: # wait for the ESC key to exit
    cv2.destroyAllWindows()
elif key == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('images/dolphin_3.png', img)
    cv2.destroyAllWindows()


################################################# DISPLAY IMAGES WITH MATPLOTLIB
print 'Display Images using Matplotlib: \n'

# display a B+W Image
import matplotlib.pyplot as plt

img = cv2.imread('images/dolphin.png')
plt.title('Monochormatic Images in Matplotlib')
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([]) # hide x and y tick values
plt.show()

# get the image size
height, width = img.shape[:2]
print 'Image Width: {}px, Image Height: {}px'.format(width, height)
print 'Image Type: {}'.format(type(img)) # openCV stores images as np.ndarray

# *WARNING*: Color images loaded by OpenCV is in BGR mode. But Matplotlib
# displays in RGB mode. So color images will not be displayed correctly in
# Matplotlib if image is read with OpenCV.

# *WRONG*
# Example of how matplotlib displays color images from OpenCV incorrectly
img_color = cv2.imread('images/fruit.png')
plt.title('How OpenCV images (BGR) display in Matplotlib (RGB)')
plt.imshow(img_color), plt.xticks([]), plt.yticks([])
plt.show()

# *CORRECT*
# Option #1 convert the color using cv2.COLOR_BGR2RGB
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
plt.title('Correct Display after converting with cv2.COLOR_BGR2RGB')
plt.imshow(img_rgb), plt.xticks([]), plt.yticks([])
plt.show()
