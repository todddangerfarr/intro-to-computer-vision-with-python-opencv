###############################################################################
#                   BASICS 0.5: Border Types in OpenCV                        #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

###################################################################### OVERVIEW
# Border types will be very important when filtering images as in order to keep
# the resulting image the same size because the filters extend over the edges of
# the image.  This .py file explores the various border types

# read in image
img = cv2.imread('images/leaves.png')
red = [255, 0, 0] # contant border color


################################################################### BORDER TYPES
# border types and a breif description
#
# cv2.BORDER_REPLICATE - Last element is replicated throughout for the total
# width specified, for example:  11111|1234567|777777
#
# cv2.BORDER_REFLECT - Border will be mirror reflection of the elements near
# the edges, for example : 54321|1234567|76543
#
# cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT - Similar to cv2.BORDER_REFLECT
# but, the value at the edge is not refelcted, for example: 65432|1234567|65432
#
# cv2.BORDER_WRAP - Pulls the border from the opposite edge. It will look like
# this : 34567|1234567|12345
#
# cv2.BORDER_CONSTANT - Adds a constant colored border. The value in RGB or BGR
# should be given as next argument. In this case Red [255, 0, 0]
replicate = cv2.copyMakeBorder(img,75,75,75,75,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img,75,75,75,75,cv2.BORDER_REFLECT)
reflect_101 = cv2.copyMakeBorder(img,75,75,75,75,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img,75,75,75,75,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img,75,75,75,75,cv2.BORDER_CONSTANT, value=red)

# because matplotlib expects RGB images and openCV reads as BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
replicate_rgb = cv2.cvtColor(replicate, cv2.COLOR_BGR2RGB)
reflect_rgb = cv2.cvtColor(reflect, cv2.COLOR_BGR2RGB)
reflect_101_rgb = cv2.cvtColor(reflect_101, cv2.COLOR_BGR2RGB)
wrap_rgb = cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB)
constant_rgb = cv2.cvtColor(constant, cv2.COLOR_BGR2RGB)

# plot the images with borders
plt.subplot(231), plt.imshow(img_rgb), plt.title('ORIGINAL')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(replicate_rgb), plt.title('REPLICATE')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(reflect_rgb), plt.title('REFLECT')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(reflect_101_rgb), plt.title('REFLECT_101')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(wrap_rgb), plt.title('WRAP')
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(constant_rgb), plt.title('CONSTANT')
plt.xticks([]), plt.yticks([])

plt.show()

# NOTE: The Reflection is prefered if you want to keep the image the same around
# the edges.  This is because it's the most likely to avoid a hard boundary.
