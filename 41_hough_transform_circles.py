###############################################################################
#                  MODEL FITTING 4.1 Hough Transform Circles                  #
#                               By: Todd Farr                                 #
###############################################################################

# A circle can be represented parametrically by the following equation:
#
#        (x - x_center)**2 + (y - y_center)**2 = r**2
#
# Where x_center, y_center is the point is space that is the center of the
# circle.  Because it has 3 parameters as oppossed to a line that only has 2,
# we would need a 3-demensional Hough Space in this instance.  With the added
# demension the size of our Hough Table increases exponetially and becomes
# quite inefficient.  The OpenCV HoughCircles() Class however uses the Gradient
# information from the edges to help alieviate this problem.
#
# HoughCircles() -> Finds circles in a grayscale image using Hough Transform
# cv2.HoughCircles(image, method, dp, minDist, param1,param2,minRadius,maxRadus)
# image    --> 8-bit, single channel, grayscale input image
# method   --> Detection method to use.
#### Currently, the only implemented method is CV_HOUGH_GRADIENT
# dp       --> Inverse ratio of the accumulator resolution to the image
#### resolution For example, if dp=1 , the accumulator has the same resolution
#### as the input image. If dp=2 , the accumulator has half as big width and
#### height.
# minDist  --> Minimum distance between the centers of the detected circles. If
#### the parameter is too small, multiple neighbor circles may be falsely
#### detected in addition to a true one. If it is too large, some circles may be
#### missed.
# param1   --> First method-specific parameter. In case of CV_HOUGH_GRADIENT, it
#### is the higher threshold of the two passed to the Canny() edge detector (the
#### lower one is twice smaller).
# param2   --> Second method-specific parameter. In case of CV_HOUGH_GRADIENT,
#### it is the accumulator threshold for the circle centers at the detection
#### stage. The smaller it is, the more false circles may be detected. Circles,
#### corresponding to the larger accumulator values, will be returned first.
# minRadius --> Minimum circle radius
# maxRadius --> Maximum circle radius

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read in and show original image
coins = cv2.imread('images/coins.png')
cv2.imshow('Original Coins Image', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert image to grayscale
coins_grayscale = cv2.cvtColor(coins, cv2.COLOR_RGB2GRAY)

# blur prior to canny edge detection to remove some noise and clean up edges
coins_blur = cv2.GaussianBlur(coins, (5, 5), 1)

# Canny Edge Detection and show image
coins_canny = cv2.Canny(coins_blur, 110, 200)
cv2.imshow('Canny Edge Detection', coins_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


############################################ HOUGH TRANSFORM (CIRCLES) IN OPENCV
# find circles using OpenCV built in HoughCircles Class.
circles = cv2.HoughCircles(coins_canny, method=cv2.HOUGH_GRADIENT, dp=1,
    minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(coins, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw center point
    cv2.circle(coins, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Detected Circles', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()
