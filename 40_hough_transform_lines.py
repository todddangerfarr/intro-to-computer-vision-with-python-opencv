###############################################################################
#                  MODEL FITTING 4.0 Hough Transform Lines                    #
#                               By: Todd Farr                                 #
###############################################################################

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# The Hough Transform is a popular algorithm for detecting any shape that can
# be represented in a parametric mathmatical form in binary images. This
# usually means that images need to be thresholded or filtered prior to running
# the Hough Transform.

# read in shapes image and convert to grayscale
shapes = cv2.imread('images/shapes.png')
cv2.imshow('Original Image', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)

# blur image (this will help clean up noise for Canny Edge Detection)
# see Chapter 2.0 for Guassian Blur or check OpenCV documentation
shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

# find Canny Edges and show resulting image
canny_edges = cv2.Canny(shapes_blurred, 100, 200)
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################################ HOUGH FROM SCRATCH USING NUMPY
# Step 1: Corner or Edge Detection for a Binary Image (see above)
print canny_edges[50,:] # print one row of the image to show binary

def hough_lines_accumulator(img, distance_resolution_pixels=1):
    ''' Returns a hough line accumulator matrix for the given binary image. '''
    #Step 2: Create the rhos and Theta ranges for creating the Hough Space.
    height, width = img.shape
    img_diag = np.ceil(np.sqrt( width**2 + height**2)) # get image diagnal
    rhos = np.arange(-img_diag, img_diag + 1, distance_resolution_pixels)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Step 3: create the empty Hough space (accumulator)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_indicies, x_indicies = np.nonzero(img) # row and column indexes for edges

    # loop through pixels from binary edge image
    for pixel_index in range(len(x_indicies)):
        x = x_indicies[pixel_index]
        y = y_indicies[pixel_index]

        # nested loop though theta values
        for theta_index in range(len(thetas)):
            # Calculate rho value image diagonal is added for a positive index
            rho = int((x * np.cos(thetas[theta_index]) +
                       y * np.sin(thetas[theta_index])) + img_diag)
            # Step 4: increment accumulator at rho and theta_index (voting)
            accumulator[rho, theta_index] += 1

    return accumulator

shapes_accumulator = hough_lines_accumulator(canny_edges)

fig = plt.figure(figsize=(6, 10))
fig.canvas.set_window_title('Hough Accumulator Plot')
plt.imshow(shapes_accumulator, cmap='gray')
plt.xlabel('Theta Values'), plt.ylabel('Rho Values')
plt.tight_layout()

plt.show()

# Step 5: Peak finding.  Need to write a funcion that finds the peaks
print np.max(shapes_accumulator), np.min(shapes_accumulator)

############################################# HOUGH TRANSFORM (LINES) IN OPENCV
# apply the hough transform
# cv2.HoughLines(image, rho, theta, threshold, srn, stn)
# image     --> 8-bit single-channel binary source image (canny edges image)
# rho       --> Distance resolution of the accumulator in pixels
# theta     --> Angle resolution of the accumulator in radians
# threshold --> Accumulator parameter threshold (Min # of votes for return)

# Detect major lines and mark them in Red
red = (0, 0, 255) # line colors
lines = cv2.HoughLines(canny_edges, 1, np.pi/180, 120)

# cycle through all rho and theta returned by cv2.HoughLines() and convert
# them back to the image space
for i in range(0, len(lines)): # for line in lines
    rho, theta = lines[i][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    # multiple a & b by large enough scalars to ensure lines cross image
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(shapes, (x1, y1), (x2, y2), red, 2)

cv2.imshow('Identify Major Lines with OpenCV Built-ins', shapes)
cv2.waitKey(0)
cv2.destroyAllWindows()
