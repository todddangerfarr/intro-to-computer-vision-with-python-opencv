###############################################################################
#          PERSPECTIVE IMAGING 5.0: Depth Mapping from Stereo Images          #
#                                By: Todd Farr                                #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

img_left = cv2.imread('images/astro_left.png', 0)
img_right = cv2.imread('images/astro_right.png', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img_left, img_right)
plt.imshow(disparity, cmap='gray')
plt.show()
