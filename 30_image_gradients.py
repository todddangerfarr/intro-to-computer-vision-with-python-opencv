###############################################################################
#                   EDGE DETECTION 3.0: Image Gradients                       #
#                              by: Todd Farr                                  #
#                                                                             #
# OpenCV provides three types of gradient filters (often referred to as high- #
# pass filters). These are Sobel, Scharr and Laplacian.  This .py file        #
# explores their use and their differences.                                   #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load image
bicycle_img = cv2.imread('images/bicycle.png', 0)

########################################################### SOBEL EDGE DETECTION
# Sobel Filtering for edge detection uses two 3X3 kernels which are then
# convolved with the original image to calculate approximations of the
# gradients (derivatives). One kernel is for detecting horizontal changes, while
# the other is for vertical changes.
#
#                          | -1  0  +1 |                         | +1  +2  +1 |
#  x-dir kernel =  1 / 8 * | -2  0  +2 |,  y-dir kernl = 1 / 8 * |  0   0   0 |
#                          | -1  0  +1 |                         | -1  -2  -1 |
#
# cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta, borderType=DEFUALT_BORDER)
# src         --> the input image
# ddepth      --> the depth of the output image (see below for options)
##### if src.depth() = CV_8U,           ddepth = -1 / CV_16S / CV_32F / CV_64F
##### if src.depth() = CV_16U / CV_16S, ddepth = -1 / CV_32F / CV_64F
##### if src.depth() = CV_32F,          ddepth = -1 / CV_32F / CV_64F
##### if src.depth() = CV_64F,          ddepth = -1 / CV_64F
# xorder     --> order of the derivative x
# yorder     --> order of the derivative y
# ksize      --> size of the kernel has to be 1, **3, 5 or 7
# scale      --> scale factor for the computed derivative values
# delta      --> delta value that is added to the results
# borderType --> pixel extrapolation method

# calculate sobel in the x and y direction for the image
sobelx = cv2.Sobel(bicycle_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(bicycle_img, cv2.CV_64F, 0, 1, ksize=5)


########################################################## SCHARR EDGE DETECTION
# cv2.Scharr(src, ddepth, dx, dy, scale, delta, borderType=DEFUALT_BORDER)
# src        --> the input image
# ddepth     --> the depth of the output image (see above for options)
# dx         --> order of the derivative x
# dy         --> order of the derivative y
# scale      --> scale factor for the computed derivative values
# delta      --> delta value that is added to the results
# borderType --> pixel extrapolation method

# calculate scharr in the x and y directions for the image
scharrx = cv2.Scharr(bicycle_img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(bicycle_img, cv2.CV_64F, 0, 1)


####################################################### LAPLACIAN EDGE DETECTION
# cv2.Laplacian(src, ddepth, ksize, scale, delta, borderType=DEFAULT_BORDER)
# src        --> the input image
# ddepth     --> Desired depth of the output image (see above for options)
# ksize      --> Apature size used to compute secon-derivative filters
# scale      --> Option scale factor for the computed Laplacian values
# delta      --> value that's added to the result prior to output
# borderType --> Pixel extrapolation method around the edges.

# apply the laplacian filter to the image using OpenCV
laplacian = cv2.Laplacian(bicycle_img, cv2.CV_64F)

# show the original, sobel, scharr and
plt.figure(figsize=(12, 8))
plt.subplot(231),plt.imshow(bicycle_img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(sobely, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# scharr
plt.subplot(234),plt.imshow(scharrx, cmap = 'gray')
plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(scharry, cmap = 'gray')
plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])

# laplacian
plt.subplot(236),plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.suptitle('Sobel, Scharr & Laplacian Edge Detection', fontsize=16)
plt.tight_layout()
plt.show()


############################################################ GRADIENT DIRECTIONS
# read in image & convert to float normalized 0-1 similar to Matlab im2double()
# This is a work in progress --> need to figure out imgradient equivalent in
# OpenCV
img = cv2.imread('images/octogon.png')
img_normal = img.astype('float') / 255

sobelx = cv2.Sobel(img_normal, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img_normal, cv2.CV_64F, 0, 1)

plt.subplot(121), plt.imshow(img_normal)
plt.subplot(122), plt.imshow(sobely)
plt.show()
