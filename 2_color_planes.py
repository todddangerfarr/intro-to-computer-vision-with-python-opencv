# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read and display image
img = cv2.imread('images/fruit.png')
cv2.imshow('Fruit Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# because the image has been loaded in color it's size is 3 numbers
# height, width and number of channels
height, width, channels = img.shape[:3]
print 'Image height: {}, Width: {}, # of channels: {}'.format(height, width, channels)

# openCV reads as BGR mode
fruit_blues = img[:, :, 0]
fruit_greens = img[:, :, 1]
fruit_reds = img[:, :, 2]

# show blue, green and red image planes for the fruit image
cv2.imshow('Fruit Blues', fruit_blues)
cv2.imshow('Fruit Greens', fruit_greens)
cv2.imshow('Fruit Reds', fruit_reds)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plot values for each color plane on a specific row
plt.plot(fruit_blues[49, :], color='blue')
plt.plot(fruit_greens[49, :], color='green')
plt.plot(fruit_reds[49, :], color='red')
plt.show()
