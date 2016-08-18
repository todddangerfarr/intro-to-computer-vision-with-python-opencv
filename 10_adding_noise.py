###############################################################################
#                    NOISE 1.0: Adding Noise to Images                        #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load and display original image
img = cv2.imread('images/saturn.png', 0)
cv2.imshow('Original Saturn Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################################# ADD GAUSSIAN NOISE
# first create an empty np array full of zeros the same size as the image
im = np.zeros(img.shape[:2], np.uint8) # do not use original image it overwrites it
mean = 0 # the gaussian mean

# effects of sigma (standard deviation (bell curve spread)) on noise
for i, sigma in enumerate(range(5, 51, 5)):
    gaussian = cv2.randn(im, mean, sigma) # create the random distribution
    saturn_gauss = cv2.add(img, gaussian) # add the noise to the original image
    plt.subplot(2, 5, i + 1), plt.imshow(saturn_gauss, cmap='gray')
    plt.title('Sigma = {}'.format(sigma)), plt.xticks([]), plt.yticks([])
plt.show()

# You can add noise to individual channels of color images by declaring the
# Gaussian mean and sigma as tuples of 3 values (B, G, R) for the blue, green
# and red channels.  You also need to make sure you use the full image.shape, do
# not slice it using [:2]


######################################################## ADD SALT & PEPPER NOISE
moon = cv2.imread('images/moon.png', 0)
cv2.imshow('Original Moon Image', moon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# salt and peppering manually
rows, cols = moon.shape
salt_vs_pepper_ratio = 0.5
amount = 0.007
moon_salted_and_peppered = moon.copy()
num_salt = np.ceil(amount * moon.size * salt_vs_pepper_ratio)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in moon.shape]
moon_salted_and_peppered[coords] = 255
num_pepper = np.ceil(amount * moon.size * (1 - salt_vs_pepper_ratio))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in moon.shape]
moon_salted_and_peppered[coords] = 0

# show salt and peppered image
cv2.imshow('Salt & Peppered Moon', moon_salted_and_peppered)
cv2.waitKey(0)
cv2.destroyAllWindows()
