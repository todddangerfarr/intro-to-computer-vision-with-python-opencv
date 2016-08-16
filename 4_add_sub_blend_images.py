# imports
import numpy as np
import cv2

# read in two image files
bicycle = cv2.imread('images/bicycle.png')
dolphin = cv2.imread('images/dolphin.png')

# show original images
cv2.imshow('Bicycle Image', bicycle)
cv2.imshow('Dolphin Image', dolphin)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *Note*: There is a difference between OpenCV addition and Numpy addition.
# OpenCV addition is a saturated operation while Numpy addition is a modulo
# operation, for example:
x = np.uint8([250])
y = np.uint8([10])
print 'Open CV Addition {}'.format(cv2.add(x, y)) # 250+10 = 260 => 255
print ''
print 'Numpy Addition {}'.format(x+y) # 250+10 = 260 % 256 = 4

# the need to be the same size for pairwise addition or the second img can just
# be a scalar value
if bicycle.shape[:2] == dolphin.shape[:2]:
    sum_img = cv2.add(bicycle, dolphin) # add images together
    cv2.imshow('Summed Images', sum_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    average_img = bicycle / 2 + dolphin / 2
    alt_average_img = cv2.add(bicycle, dolphin) / 2
    cv2.imshow('Averaged Images', average_img)
    cv2.imshow('Alt. Averaged Images', alt_average_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# SUBTRACTING IMAGES via the cv2.absdiff fuction
# order does not matter with absdiff
diff = cv2.absdiff(bicycle, dolphin)
cv2.imshow('Subtracted Images', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ALPHA BLENDING IMAGES via the blending function cv2.addWeighted
# g(x) = (1 - alpha)f_0(x) + alpha f_1(x)
# By varying alpha from 0 to 1, you can perform a cool transition between one
# image to another.
# dst = alpha * img1 + beta * img2 + gamma
dst = cv2.addWeighted(bicycle, 0.25, dolphin, 0.75, 0)
cv2.imshow('Blended Images', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
