###############################################################################
#                  EDGE DETECTION 3.1 Canny Edge Detection                    #
#                                By: Todd Farr                                #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt


# The Canny edge detection algorithm is a multi-stage algorithm developed by
# John F. Canny in 1986.
#
# Stage 1: noise reduction
# This is because edge detection is susceptible to noise in the image due to
# the fact that edge detection requires derivatives and noise results in local
# min and max values.  In OpenCV the noise is removed with a 5x5 Gaussian
# Filter. See Chapter 2.0 for more information.
#
# Stage 2: Finding Intensity Gradient of the image by Thresholding
# The smoothed image is filtered with a Sobel Kernel in both x and y directions
# to get Gx and Gy (the first derivatives in these directions).  From these two
# resulting images the edge gradient (magnitude) and direction (angle) can be
# calcuated:
#
#           Edge Gradient (G) = sqrt(Gx**2 + Gy**2) -> magnitude
#           Angle (theta) = arctan2(Gy / Gx)
#
# The graident direction is always perpendicular to the edges and is rounded to
# one of four angles represented by vertical, horizontal or 45 deg diagnols.
#
# Stage 3: Thin the Edges by Non-maximum suppression
# Every pixel is checked to see if it's the maximum within it's neigborhood in
# the direction of the graident. The pixel is suppressed if it's not the
# maximum value.  The result is a binary image with "thin edges."
#
# Stage 4: Connect pixels to create contours by Hysteresis Thresholding.
# This is the stage for determining which pixels are really edges and which are
# not.  For this stage we need thresholding values (minValue and MaxValue). Any
# values above the MaxValue will be considered sure-edges, where values between
# these thresholds will only be included if they are connected to lines that
# extend above the Max threshold value. This stage also removes pixel noise
# because our assumptions are that edges are long lines.


################################################ CANNY EDGE DETECTION EXAMPLE 1
# Find the hidden message in two images
# read in images as grayscale, second argument set to 0 in OpenCV imread()
frizzy = cv2.imread('images/frizzy.png', 0)
frommer = cv2.imread('images/frommer.png', 0)

# Canny() edge detection in OpenCV
# cv2.Canny(image, threshold1, threshold2, aperatureSize, L2gradient)
# image        --> input image
# threshold1   --> first threshold for hysteresis procedure
# threshold2   --> second threshold for hysteresis procedure
# apertureSize --> aperture size for the Sobel() operator
# L2gradient   --> a flag indicating whether (default=false):
#### true:  l2_norm = sqrt((dI/dx)**2 + (dI/dy)**2) [more accurate]
#### false: l2_norm = |dI/dx| + |dI/dy|
frizzy_edges = cv2.Canny(frizzy, 100, 200)
frommer_edges = cv2.Canny(frommer, 100, 200)

# find common edge pixels, because Canny returns a binary image, we can used
# bitwise_and from OpenCV to return the resulting image of shared pixels
# between the edges from frizzy and frommer revealling the secret code.
common_edge_pixels = cv2.bitwise_and(frizzy_edges, frommer_edges)

# show all images using matplotlib
plt.figure(figsize=(10, 10))
plt.subplot(321), plt.imshow(frizzy, cmap='gray')
plt.title('Frizzy Original'), plt.xticks([]), plt.yticks([])
plt.subplot(322), plt.imshow(frizzy_edges, cmap='gray')
plt.title('Frizzy Edges'), plt.xticks([]), plt.yticks([])

plt.subplot(323), plt.imshow(frommer, cmap='gray')
plt.title('Frommer Original'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.imshow(frommer_edges, cmap='gray')
plt.title('Frommer Edges'), plt.xticks([]), plt.yticks([])

plt.subplot(325), plt.imshow(common_edge_pixels, cmap='gray')
plt.title('Secret Message - Common Pixels'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


################################################ CANNY EDGE DETECTION EXAMPLE 2
# The Lena image is a really common image for Canny edge Detection
# read in the image
lena = cv2.imread('images/lena.png')

# convert image to grayscale (NOTE: we could have used 0 as arg 2 for imread())
lena_grayscale = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)

# find canny edges
lena_canny = cv2.Canny(lena_grayscale, 100, 200)
cv2.imshow('Lena Canny Edge Detection', lena_canny)
cv2.waitKey()
cv2.destroyAllWindows()


#################################################### EFFECTS OF LOWER THRESHOLD
# Exploring the effects of the lower threshold value with Canny Detection
fig = plt.figure(figsize=(20, 10))
fig.canvas.set_window_title(
    'The Effects of the Lower Threshold Value on Canny Edge Detection')
for i, value in enumerate(range(10, 181, 10)):
    canny = cv2.Canny(lena_grayscale, value, 200)
    plt.subplot(3, 6, i + 1), plt.title('Lower Threshold = {}'.format(value))
    plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


#################################################### EFFECTS OF UPPER THRESHOLD
# Exploring the effects of the upper threshold value with Canny Detection
fig = plt.figure(figsize=(20, 10))
fig.canvas.set_window_title(
    'The Effects of the Upper Threshold Value Canny Edge Detection')
for i, value in enumerate(range(210, 381, 10)):
    canny = cv2.Canny(lena_grayscale, 100, value)
    plt.subplot(3, 6, i + 1), plt.title('Upper Threshold = {}'.format(value))
    plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


####################################################### EFFECTS OF APATURE SIZE
# Exploring the effects of apature size with Canny Edge Detection
# Remember this is this aperture size for computing the gradients in the Sobel
# opeator. Therefore in OpenCV Sobel(), this would be ksize and so our only
# choices indicated by the documentation are 3, 5 & 7
fig = plt.figure(figsize=(10, 10))
fig.canvas.set_window_title(
    'The Effects of the Aperture Size Canny Edge Detection')
for i, value in enumerate(range(3, 8, 2)):
    canny = cv2.Canny(lena_grayscale, 100, 200, apertureSize = value)
    plt.subplot(2, 2, i + 1), plt.title('Aperture Size = {}'.format(value))
    plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


############################################################## EFFECTS OF SIGMA
# Unfortunately, the OpenCV Canny function doesn't let you change the filter
# kernel it uses via the function parameters. However. You can generate the
# same results by first blurring the input image, and then passing this blurred
# image into the Canny function.
#
# Since both the Gaussian Blur and Sobel filters are linear, passing a blurred
# input image to the OpenCV Canny() function is mathematically equivalent to
# what Matlab does because of the principle of superposition.
# (NOTE: *This assumes this is the convolution operator)
#
# The Matlab method: the sobel and blur operations are combined into
# a single filter, and that filter is then convolved with the image
#### matlabFancyFilter = (sobel * blur);
#### gradient = matlabFancyFilter * image;
#
# Equivalent method: image is first convolved with the blur filter, and
# then convolved with the sobel filter.
#### gradient = sobel * (blur * image); // image is filtered twice


# Testing Sigma values between 0.5 - 2 with a constant filter size (7 x 7)
fig = plt.figure(figsize=(10, 10))
fig.canvas.set_window_title(
    'The Effects of the Sigma (GaussianBlur) on Canny Edge Detection')
for i, value in enumerate(range(1, 5)):
    smoothed_lena = cv2.GaussianBlur(lena_grayscale, (7, 7), value / 2.)
    canny = cv2.Canny(smoothed_lena, 100, 200)
    plt.subplot(2, 2, i + 1), plt.title('Sigma = {:.1f}'.format(value / 2.))
    plt.imshow(canny, cmap='gray'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
