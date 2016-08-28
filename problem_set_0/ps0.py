# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load images, remember openCV loads color images as BGR
waterfall_original = cv2.imread('input/waterfall.png')
mountains_original = cv2.imread('input/mountains.png')

####### Problem - swapping blue and red color planes
waterfall_blues = waterfall_original[:,:,0] # access blue color plane
waterfall_reds = waterfall_original[:,:,2]  # access the red color plane

# show image planes for reference
cv2.imshow('Red Plane', waterfall_reds)
cv2.imshow('Blue Plane', waterfall_blues)
cv2.waitKey(0)
cv2.destroyAllWindows()

# clone and swap color planes
# in OpenCV C++ you would use the .clone() method, however to copy and image in
# OpenCV Python us np.copy() from numpy
waterfall_red_blue_swap = waterfall_original.copy()
waterfall_red_blue_swap[:,:,0] = waterfall_reds
waterfall_red_blue_swap[:,:,2] = waterfall_blues

# show original and R-B swapped color planes
cv2.imshow('Original Image', waterfall_original)
cv2.imshow('Red & Blue Swapped', waterfall_red_blue_swap)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save red and blue swapped image
cv2.imwrite('output/waterfall_red_blue_swapped.png', waterfall_red_blue_swap)


####### Problem - create a green and red monochromatic image
waterfall_greens = waterfall_original[:,:,1] # access the green color plane
cv2.imshow('Green Monochromatic Image', waterfall_greens)
cv2.imshow('Red Monochromatic Image', waterfall_reds)
cv2.waitKey(0)
cv2.destroyAllWindows()

# because green "looks" better we will save this as the monochromatic version
cv2.imwrite('output/water_monochromatic.png', waterfall_greens)
waterfall_mono = waterfall_greens.copy()


###### Problem - Replacement of Pixels
# replace the center 100 x 100 pixel square of the mountains mono image with the
# 100 x 100 center pixel square from the waterfall mono image
# first see what mono channel looks better for the mountains image
mountains_blues = mountains_original[:,:,0]
mountains_greens = mountains_original[:,:,1]
mountains_reds = mountains_original[:,:,2]

# show images
cv2.imshow('Blue Monochromatic Image', mountains_blues)
cv2.imshow('Green Monochromatic Image', mountains_greens)
cv2.imshow('Red Monochromatic Image', mountains_reds)
cv2.waitKey(0)
cv2.destroyAllWindows()

# green is the winner, copy it
mountains_mono = mountains_greens.copy()

# find the chunk
waterfall_center = np.array(waterfall_mono.shape[:2]) / 2.
waterfall_center_chunk = waterfall_mono[
        int(waterfall_center[0] - 50):int(waterfall_center[0] + 50),
        int(waterfall_center[1] - 50):int(waterfall_center[1] + 50)
    ]
mountains_center = np.array(mountains_mono.shape[:2]) / 2.
mountains_mono_center_replaced = mountains_mono.copy() # copy first
mountains_mono_center_replaced[
        int(mountains_center[0] - 50):int(mountains_center[0] + 50),
        int(mountains_center[1] - 50):int(mountains_center[1] + 50)
    ] = waterfall_center_chunk

# show center chunk replaced mono moutnains image and save on close
cv2.imshow('Center Replaced', mountains_mono_center_replaced)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/mountains_mono_center_replaced.png',
    mountains_mono_center_replaced)


###### Problem - Arithmetic and Geometric Operations
print 'The minimum value in the mono waterfall image is:      {}'.format(
    waterfall_mono.min())
print 'The maximum value in the mono waterfall image is:      {}'.format(
    waterfall_mono.max())
print 'The average value in the mono waterfall image is:      {:.2f}'.format(
    waterfall_mono.mean())
print 'The standard deviation in the mono waterfall image is: {:.2f}'.format(
    waterfall_mono.std())

# subtract the mean, then divide by the standard deviation, then multiply by 10
# finally add the mean back in
waterfall_arithmetic = waterfall_mono.copy()
waterfall_arithmetic = cv2.absdiff(waterfall_arithmetic, waterfall_arithmetic.mean())
waterfall_arithmetic = cv2.divide(waterfall_arithmetic, waterfall_arithmetic.std())
waterfall_arithmetic = cv2.multiply(waterfall_arithmetic, 10)
waterfall_arithmetic = cv2.add(waterfall_arithmetic, waterfall_arithmetic.mean())
cv2.imshow('Waterfall Arithmetic', waterfall_arithmetic)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/waterfall_arithmetic.png', waterfall_arithmetic)

# shift the waterfall_greens image 2 pixels to the left
M = np.float32([[1,0,2],[0,1,0]]) # the transformation matrix for Translation
rows, cols = waterfall_greens.shape[:2]
waterfall_greens_shifted = cv2.warpAffine(waterfall_greens, M, (cols, rows))
waterfall_greens_sub_shifted = cv2.subtract(
    waterfall_greens, waterfall_greens_shifted)
cv2.imshow('Waterfall Greens Sub Shifted', waterfall_greens_sub_shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/waterfall_greens_sub_shifted.png', waterfall_greens_sub_shifted)


###### Problem - Noise
# compare Gaussian Noise between blue and green channels in Waterfall image
im = np.zeros(waterfall_original.shape, np.uint8) # do not use original image it overwrites it
mean = (0, 1, 0) # gaussian mean BGR channels
sigma = (0, 10, 0) # gaussian sigma BGR channels
gaussian_noise = cv2.randn(im, mean, sigma)
waterfall_noise_green = cv2.add(waterfall_original, gaussian_noise)
cv2.imshow('Waterfall Green Noise', waterfall_noise_green)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/waterfall_noise_green.png', waterfall_noise_green)

# repeat for Blue Channel
mean = (1, 0, 0) # gaussian mean only for the blue channel (BGR)
sigma = (10, 0, 0) # gaussian sigma only for the blue channel (BGR)
gaussian_noise = cv2.randn(im, mean, sigma)
waterfall_noise_blue = cv2.add(waterfall_original, gaussian_noise)
cv2.imshow('Waterfall Blue Noise', waterfall_noise_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output/waterfall_noise_blue.png', waterfall_noise_blue)
