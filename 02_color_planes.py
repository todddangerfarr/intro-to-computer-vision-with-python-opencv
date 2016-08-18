###############################################################################
#      BASICS 0.2: Examining color planes and how OpenCV handles them         #
#                               by: Todd Farr                                 #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

# read and display the image as a reference
img = cv2.imread('images/fruit.png')
cv2.imshow('Fruit Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###################################################### UNDERSTANDING IMAGE SHAPE
# Because the image has been loaded in color it's size (shape) is 3 numbers
# height (# of rows), width (# of cols) and finally color planes (BGR)
height, width, channels = img.shape[:3]
print 'Image height: {}, Width: {}, # of channels: {}'.format(height, width, channels)


#################################################### SHOW DIFFERENT COLOR PLANES
# Remember openCV reads as BGR mode, therefore channel 0 is blue, channel 1 is
# green and channel 2 is red
fruit_blues = img[:, :, 0]
fruit_greens = img[:, :, 1]
fruit_reds = img[:, :, 2]

# show blue, green and red image planes for the fruit image using imshow()
cv2.imshow('Fruit Blues', fruit_blues)
cv2.imshow('Fruit Greens', fruit_greens)
cv2.imshow('Fruit Reds', fruit_reds)
cv2.waitKey(0)
cv2.destroyAllWindows()


#################################### EXAMINE COLOR PLANE VALUES FOR A SINGLE ROW
# plot values for each color plane on a specific row
fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# original image
ax0 = plt.subplot(gs[0])
ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # need to convert BGR to RGB
ax0.axhline(50, color='black') # show the row being used
ax0.axvline(100, color='k'), ax0.axvline(225, color='k') # ref lines

# image slice
ax1 = plt.subplot(gs[1])
ax1.plot(fruit_blues[49, :], color='blue')
ax1.plot(fruit_greens[49, :], color='green')
ax1.plot(fruit_reds[49, :], color='red')
ax1.axvline(100, color='k', linewidth=2), ax1.axvline(225, color='k', linewidth=2)

plt.suptitle('Examining Color Plane Values for a Single Row.')
plt.show()
