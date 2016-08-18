###############################################################################
#               NOISE 1.1: Removing Noise with Gaussian Blur                  #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/saturn.png', 0)

############################################## REMOVING NOISE WITH GAUSSIAN BLUR
# fist create a noisy image
# create an empty np array full of zeros the same size as the image
im = np.zeros(img.shape[:2], np.uint8) # do not use original image it overwrites it
mean = 0 # the gaussian mean
sigma_noise = 30 # the gaussian sigma
gaussian = cv2.randn(im, mean, sigma_noise) # create the random distribution
img_noisy = cv2.add(img, gaussian) # add the noise to the original image

# clean up the noise
gauss_noise_remv = cv2.GaussianBlur(img_noisy, (21, 21), 2)

# plot original, noisy and gaussian noise removal together
plt.figure(figsize=(10, 8))
plt.subplot(131), plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_noisy, cmap='gray', interpolation='bicubic')
plt.title('Noisy'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(gauss_noise_remv, cmap='gray', interpolation='bicubic')
plt.title('Gaussian Noise Removal'), plt.xticks([]), plt.yticks([])
plt.show()
