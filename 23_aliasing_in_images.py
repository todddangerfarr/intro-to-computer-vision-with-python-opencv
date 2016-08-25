###############################################################################
#     FILTERING 2.3: Gaussians for antialiasing when downsampling images      #
#                                By: Todd Farr                                #
###############################################################################

# Aliasing in images occurs when the frequency is higher than the number of
# pixels we have to represent that frequency in the image.  This results in
# a misidentification of a signal frequency which introduces distortion or error

# In particular this Python file shows how Guassian filtering can be used to
# help solve this problem by removing the high frequencies prior to downsampling

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read in and show the image
img = cv2.imread('images/van_gogh.png')
cv2.imshow('Van Gogh Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# First create a downsampling fucntion for the image
def downsample(img, reduction_step=2):
    height, width, channels = img.shape
    rows_keep = range(0, height, reduction_step)
    cols_keep = range(0, width, reduction_step)

    ds_img = np.zeros((len(rows_keep), len(cols_keep), channels), dtype=np.uint8)
    for i, row_value in enumerate(rows_keep):
        for j, col_value in enumerate(cols_keep):
            for k in range(channels):
                ds_img[i, j, k] = img[row_value, col_value, k]

    return ds_img

# show images 
vg_half_size = downsample(img, 2)
vg_quarter_size = downsample(img, 4)
cv2.imshow('1/2 Size', vg_half_size)
cv2.imshow('1/4 Size', vg_quarter_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
