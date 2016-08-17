###############################################################################
#   Tutorial 6.2 = Using a filter as a template to identify it's location     #
#                               by: Todd Farr                                 #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# write a funciton to find the X, Y location of a template in the image
# read in image and the template
tablet_img = cv2.imread('images/tablet.png', 0)
glyph = cv2.imread('images/glyph_template.png', 0)

# template matching methods in OpenCV
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# method explinations (Need to build on this)
#
# CV_TM_CCORR_NORMED:  This is the normalized cross correlation method

def find_template_2D(image, template, method):
    rows, cols = template.shape # get the shape of the template
    result = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        bottom_right = (top_left[0] + cols, top_left[1] + rows)

    return result, top_left, bottom_right, method

# find the template location using the function above
result, top_left, bottom_right, method = find_template_2D(
                                        tablet_img, glyph, cv2.TM_CCORR_NORMED)

# draw a white bonding bax around the template location
cv2.rectangle(tablet_img, top_left, bottom_right, 255, 2)

# plot the results using MatPlotLib 
plt.subplot(121),plt.imshow(result ,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(tablet_img, cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(methods[method], fontsize=16)

plt.show()
