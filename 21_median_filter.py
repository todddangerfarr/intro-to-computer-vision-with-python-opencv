###############################################################################
#     FILTERING 2.1: Applying a Median Filter to Blur and Smooth an Image     #
#                                By: Todd Farr                                #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# This type of filter is extremely effective in removing salt and pepper noise
# see Chapter 1.2 for an example of this application
# This filters by applying the median value under the lens to the center pixel,
# as a result it's not affeted as greatly as the average would be low or high
# pixels. This is because we assume that images should have smooth transitions
# to their neighboring pixels.

# load in image
saturn = cv2.imread('images/saturn.png', 0)

############################################## APPLY MEDIAN FILTER USING OPENCV
# The second argument is the aperture linear size; it must be odd and greater
# than 1, for example: 3, 5, 7
saturn_median = cv2.medianBlur(saturn, 11)

# show both images using Matplotlib
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(saturn, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(saturn_median, cmap='gray')
plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
