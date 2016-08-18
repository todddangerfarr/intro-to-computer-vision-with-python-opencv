###############################################################################
#            BASICS 0.1: Inspecting Images by slicing Operations              #
#                               by: Todd Farr                                 #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load image and show it
img = cv2.imread('images/dolphin.png', 0) # 0 signifys grayscale
cv2.imshow('Dolphin Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################################ SLICING IMAGES
# read value from row 50 column 100 (NOTE: remember zero indexed => -1)
# NOTE:  Images are just functions or a matrix of Intensity Values, therefore
# you can slice them just as you would arrays or matricies
print 'The intensity value at row 50 & column 100 is: {}'.format(img[49, 99])
print ''

# read all columns values at row 50
print 'Row 50 column values:'
print img[49, :]
print ''

# read a 'chunk' of the image
print 'Rows 101 - 103 & columns 201 - 203'
print img[100:103, 200:203]

# plot an entire rows values (50) of an image using Matplotlib
plt.plot(img[49, :])
plt.show()
