# imports
import numpy as np
import cv2

# load and display original image
img = cv2.imread('images/saturn.png', 0)
cv2.imshow('Saturn Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ADD GAUSSIAN NOISE
# first create an empty np array full of zeros the same size as the image
im = np.zeros(img.shape[:2], np.uint8) # do not use original image it overwrites it
mean = 0 # the gaussian mean

# effects of sigma (standard deviation (bell curve spread)) on noise
for sigma in range(5, 51, 5):
    gaussian = cv2.randn(im, mean, sigma) # create the random distribution
    saturn_gauss = cv2.add(img, gaussian) # add the noise to the original image
    cv2.imshow('Gaussian Noise, Sigma={}'.format(sigma), saturn_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()
