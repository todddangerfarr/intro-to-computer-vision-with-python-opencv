###############################################################################
#      BASICS 0.4: Add, subtract & blending images together in OpenCV         #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2

# read in two image files
bicycle = cv2.imread('images/bicycle.png', 0)
dolphin = cv2.imread('images/dolphin.png', 0)

# show original images
cv2.imshow('Bicycle Image', bicycle)
cv2.imshow('Dolphin Image', dolphin)
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################### DIFFERENCES BETWEEN NUMPY & OPENCV ADDITION
print 'Differences between numpy and OpenCV addition operations:\n'
# *Note*: There is a difference between OpenCV addition and Numpy addition.
# OpenCV addition is a saturated operation while Numpy addition is a modulo
# operation, for example:
x = np.uint8([250])
y = np.uint8([10])
print 'Open CV Addition {}'.format(cv2.add(x, y)) # 250+10 = 260 => 255
print ''
print 'Numpy Addition {}\n'.format(x+y) # 250+10 = 260 % 256 = 4


##################################################################### ADD IMAGES
print 'Adding images with cv2.add():\n'
# NOTE: THe images need to be the same size for pairwise addition or the second
# argument can just be a scalar value for element-wise addition
if bicycle.shape[:2] == dolphin.shape[:2]:
    sum_img = cv2.add(bicycle, dolphin) # add images together
    cv2.imshow('Summed Images', sum_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    scaled_img = cv2.add(bicycle, 50)
    cv2.imshow('Scalar Addition on Bicycle Image', scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


################################################################ SUBTRACT IMAGES
print 'Subtracting images with cv2.absdiff() [Absolute Difference]:\n'
# Using the cv2 absolute difference method (cv2.absdiff fuction) will make it
# so we don't have negative intensities, also order does not matter with absdiff
if bicycle.shape[:2] == dolphin.shape[:2]:
    diff = cv2.absdiff(bicycle, dolphin)
    cv2.imshow('Subtracted Images', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


############################################################### AVERAGING IMAGES
print 'Averaging 2 images:\n'
# Averaging can help with lossing information at the upper bounds of the image
# Here we show two ways to do so and their resulting images
if bicycle.shape[:2] == dolphin.shape[:2]:
    average_img = bicycle / 2 + dolphin / 2
    alt_average_img = cv2.add(bicycle, dolphin) / 2
    cv2.imshow('Averaged Images', average_img)
    cv2.imshow('Alt. Averaged Images', alt_average_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


########################################################## ALPHA BLENDING IMAGES
print 'Alpha blending with cv2.addWeighted():\n'
# Alpha blending via the blending function cv2.addWeighted
#### Equation: g(x) = (1 - alpha)f_0(x) + alpha f_1(x)
# By varying alpha from 0 to 1, you can perform a cool transition between one
# image to another, it's important to note that alpha and beta should add to 1.
#### dst = alpha * img1 + beta * img2 + gamma
alpha_blending = cv2.addWeighted(bicycle, 0.25, dolphin, 0.75, 0)
cv2.imshow('Alpha Blending Images', alpha_blending)
cv2.waitKey(0)
cv2.destroyAllWindows()
