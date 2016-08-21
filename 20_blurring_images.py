###############################################################################
#               FILTERING 2.0: Blurring Images with OpenCV                    #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats as st

# load and display original black and white image
# The 2nd argument 0 signifies 1 color plane (B + W), if this argument is left
# off, the image will be 3 color planes will BGR all equal values (Grayscale)
img = cv2.imread('images/saturn.png', 0)
cv2.imshow('Original Image', img)
cv2.waitKey(0) # hold processes and wait for any key press
cv2.destroyAllWindows() # destroy all open windows


####################################################### BLURRING WITH cv2.blur()
# the first argument is the img, the second argument is the size of a normalized
# box filter (w, h) below is an example of a 3 x 3 box filter
#
#            | 1  1  1 |
#  k = 1/9 * | 1  1  1 |
#            | 1  1  1 |
#
# using matplotlib to plot the two images (original/blur) side by side
box_blur = cv2.blur(img, (5, 5)) # a 5 x 5 box filter
plt.subplot(121), plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(box_blur, cmap='gray', interpolation='bicubic')
plt.title('Box Blur 5 x 5'), plt.xticks([]), plt.yticks([])
plt.show()


########################################################### MANUAL GAUSSIAN BLUR
# Creating and inspecting Gaussian Kernels Manually
def gaussian_kernel(k_size, sigma=5):
    ''' Returns a 2D Gaussian Kernel array. k_size needs to be an odd number. '''
    interval = (2 * sigma + 1.) / k_size
    x = np.linspace(-sigma - interval/2., sigma + interval/2., k_size + 1)
    k_1d = np.diff(st.norm.cdf(x))
    k_raw = np.sqrt(np.outer(k_1d, k_1d))
    k = k_raw / k_raw.sum()
    return k

# examining the effects of sigma on the resulting gaussian
gaussians = []
for sigma in range(1, 11):
    gaussians.append(gaussian_kernel(21, sigma))

for i, gauss in enumerate(gaussians):
    plt.subplot(2, 5, i + 1)# subplot number num_rows num_cols position
    plt.imshow(gauss), plt.xticks([]), plt.yticks([])
    plt.title('Sigma = ' + str(i + 1), fontsize=9)
plt.show()


###################################################### GAUSSIAN BLUR WITH OPENCV
# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
# src    --> is the original image
# ksize  --> must be a tuple of positive odd values (width, height)
# sigmaX --> is the Gaussian deviation in the X diretion
# sigmaY --> is the Gaussian deviation in the Y diretion, if sigmaY is zero,
#### it is set to be equal to sigmaX, if both sigmas are zeros, they are computed
#### from ksize.width and ksize.height
# BorderType --> pixel extrapolation method (see borderInterpolate() for details)
gaussian_blur = cv2.GaussianBlur(img, (11, 11), 1)
cv2.imshow('Gaussian Blur 11 x 11, Sigma = 1', gaussian_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
