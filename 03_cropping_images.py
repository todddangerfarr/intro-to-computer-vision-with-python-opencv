###############################################################################
#                        BASICS 0.3: Cropping Images                          #
#                              by: Todd Farr                                  #
###############################################################################

# imports
import numpy as np
import cv2

# read an image of a bicycle
img = cv2.imread('images/bicycle.png')
cv2.imshow('Hey, Sweet Bicycle!', img) # arg 1: Window Name, arg 2: image
cv2.waitKey(0)
cv2.destroyAllWindows()

###################################### BASIC CROPPING BY NUMPY SLICING OPERATION
# crop the image rows 110-310, cols 10-160 (zero indexed)
# the limits are inclusive so image size will be +1
cropped = img[109:310, 9:160]
cv2.imshow('Cropped Image', cropped)
print "press 's' to save the image as cropped_bicycle.png\n"
key = cv2.waitKey(0) # if you are using a 64-bit machine see below
# the above line should be: key = cv2.waitKey(0) & 0xFF
if key == 27: # wait for the ESC key to exit
    cv2.destroyAllWindows()
elif key == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('images/cropped_bicycle.png', img)
    cv2.destroyAllWindows()

# get the size of the cropped image
height, width = cropped.shape[:2]
print 'Cropped Width: {}px, Cropped Height: {}px'.format(width, height)
