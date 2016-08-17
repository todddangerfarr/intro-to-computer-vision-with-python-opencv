###############################################################################
#     Tutorial 6.1 Applying a Median Filter to clean Salt & Pepper Noise      #
#                                By: Todd Farr                                #
###############################################################################

# imports
import numpy as np
import cv2

# This type of filter is extremely effective in removing salt and pepper noise
# This filters by applying the median value under the lens to the center pixel,
# as a result it's not affeted as greatly as the average would be low or high
# pixels. This is because we assume that images should have smooth transitions
# to their neighboring pixels.

# load in image and add Salt and pepper noise
moon = cv2.imread('images/moon.png', 0)

# salt and peppering manually (randomly assign coords as either white or black)
rows, cols = moon.shape
salt_vs_pepper_ratio = 0.5
amount = 0.007
moon_salted_and_peppered = moon.copy()
num_salt = np.ceil(amount * moon.size * salt_vs_pepper_ratio)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in moon.shape]
moon_salted_and_peppered[coords] = 255
num_pepper = np.ceil(amount * moon.size * (1 - salt_vs_pepper_ratio))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in moon.shape]
moon_salted_and_peppered[coords] = 0

# show salt and peppered image
cv2.imshow('Salt & Peppered Moon', moon_salted_and_peppered)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################## APPLY MEDIAN FILTER USING OPENCV
# The second argument is the aperture linear size; it must be odd and greater
# than 1, for example: 3, 5, 7
moon_median = cv2.medianBlur(moon, 5)
cv2.imshow('Salt & Peppered Moon after Median Blur', moon_median)
cv2.waitKey(0)
cv2.destroyAllWindows()
