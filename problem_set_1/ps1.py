###############################################################################
#                       PROBLEM SET 1: Edges and Lines                        #
#                              By: Todd Farr                                  #
###############################################################################

# WARNING: For some reason this code doesn't run on Mac OSX
# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt 

# See the Problem set instructions in "problem-set-1-instructions.pdf"

##################################################################### QUESTION 1
# Read in the original image
img = cv2.imread('input/ps1-input0.png')

# do some edge detection, I had to blur the image prior to cv2.Canny due to the
# fact that it captures slight diagonals in the Hough Space if I didn't.
img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_grayscale, (5, 5), 1)
img_edges = cv2.Canny(img_blur, 100, 200, apertureSize=5)

# show original and edges image
cv2.imshow('Original', img)
cv2.imshow('Edges', img_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Output: Store edge image (img_edges) as ps1-1-a-1.png
cv2.imwrite('output/ps1-1-a-1.png', img_edges)


##################################################################### QUESTION 2
# Implement a Hough Transform method for finding lines.
# Step 1:  Create the Hough Line Accumulator (H)
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


def plot_hough_acc(H, H1=None, plot_title='Hough Accumulator Plot'):
    # plot the Hough Line Accumulator
    if (type(H1) == np.ndarray):
        fig = plt.figure(figsize=(8, 10))
        fig.canvas.set_window_title(plot_title)
        plt.subplot(121), plt.imshow(H, cmap='gray')
        plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
        plt.subplot(122), plt.imshow(H1, cmap='gray')
    else:
        fig = plt.figure(figsize=(6, 10))
        fig.canvas.set_window_title(plot_title)
        plt.imshow(H, cmap='gray')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


# Step 2:  Find the num highest peaks from Hough Line Accumulator
# The first attempt to find hough peaks it works but doesn't surpress values
# after a peak in that neighborhood has been found (issues with noisy images)
def hough_simple_peaks(H, num_peaks):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to local maxima. '''
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


# Second attempt at a more tunable hough_peaks algorithm
def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the Hough Space with neighborhoods deleted
    return indicies, H, H1


def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


# build Hough Accumulator from img_edges
H, rhos, thetas = hough_lines_acc(img_edges, theta_resolution=0.5)

# plot the resulting accumulator
plot_hough_acc(H)

# find the indicies for the Hough Peaks
indicies = hough_simple_peaks(H, 6)

# use these indicies to find corresponding rhos and thetas for line drawing
hough_lines_draw(img, indicies, rhos, thetas)
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


##################################################################### QUESTION 3
# Repeat question 2, but this time the image has noise
# read in image and clean up with Gaussian Blur
img_noisy = cv2.imread('input/ps1-input0-noise.png')
img_noisy_grayscale = cv2.cvtColor(img_noisy, cv2.COLOR_RGB2GRAY)
img_noisy_blur = cv2.GaussianBlur(img_noisy_grayscale, (5, 5), 6)

# edge detection and show images
img_noisy_edges = cv2.Canny(img_noisy_blur, 150, 200)
cv2.imshow('Original', img_noisy)
cv2.imshow('Blurred', img_noisy_blur)
cv2.imshow('Edges', img_noisy_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# build the Hough Line Accumulator
H, rhos, thetas = hough_lines_acc(img_noisy_edges)
# find the indicies for the Hough Peaks
indicies, H, H1 = hough_peaks(H, 8, nhood_size=20)
# plot H (Hough highlighted) and H1 (Hough neighborhoods removed)
plot_hough_acc(H, H1, plot_title='Hough Peaks Highlighted')

# use these indicies to find corresponding rhos and thetas for line drawing
hough_lines_draw(img_noisy, indicies, rhos, thetas)
cv2.imshow('Hough Lines', img_noisy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save Hough space and the noisy image with edges found
cv2.imwrite('output/ps1-3-c-1.png', H)
cv2.imwrite('output/ps1-3-c-2.png', img_noisy)


##################################################################### QUESTION 4
# Find the lines along the pens in the ps1-input1.png image
img_pens = cv2.imread('input/ps1-input1.png')

# convert to grayscale and find edges
img_pens_grayscale = cv2.cvtColor(img_pens, cv2.COLOR_RGB2GRAY)
img_pens_blur = cv2.GaussianBlur(img_pens_grayscale, (5, 5), 2)
img_pens_edges = cv2.Canny(img_pens_blur, 100, 200)
cv2.imshow('Original Image', img_pens)
cv2.imshow('Grayscale Blurred', img_pens_blur)
cv2.imshow('Canny Edges', img_pens_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save blurred grayscale image as ps1-4-a-1.png
cv2.imwrite('output/ps1-4-a-1.png', img_pens_blur)

# build the Hough Accumulator
H, rhos, thetas = hough_lines_acc(img_pens_edges, theta_resolution=0.5)
indicies, H, H1 = hough_peaks(H, 4, nhood_size=11)
plot_hough_acc(H, H1, plot_title='Hough Peaks Highlighted')

# draw lines
hough_lines_draw(img_pens, indicies, rhos, thetas)
cv2.imshow('Hough Lines', img_pens)
cv2.waitKey(0)
cv2.destroyAllWindows()


##################################################################### QUESTION 5
# This is still a work in progress
# finding Hough Circles with known radii
def hough_circles_acc(img, radius):
    ''' A function used to build a Hough Accumulator for finding circles in an
        image. '''
    height, width = img.shape
    y_idxs, x_idxs = np.nonzero(img)
    H = np.zeros((width, height), dtype=np.uint64)

    # cycle through edge pixels
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        # for each possible gradient direction
        for theta in np.deg2rad(np.arange(0, 360)):
            a = int(x - radius*np.cos(theta))
            b = int(y + radius*np.sin(theta))
            # make sure it stays within the image bounds
            if (a < width and a > 0 and b < height and b > 0):
                H[a, b] += 1

    return H, radius


def hough_circles_draw(img, indicies, radius):
    """ This function draws circles found using indicies from the Hough
        Accumulator. """
    for index in indicies:
        # draw circle(img, center location, radius, color, thickness)
        cv2.circle(img, index, radius, (0, 0, 255), 2)

H, radius = hough_circles_acc(img_pens_edges, 20)
indicies, H, H1 = hough_peaks(H, 10, nhood_size=20)
plot_hough_acc(H, H1, plot_title='Hough Circles Peaks Highlighted')

# draw circles
hough_circles_draw(img_pens, indicies, radius)
cv2.imshow('Hough Lines + Hough Circles', img_pens)
cv2.waitKey(0)
cv2.destroyAllWindows()


##################################################################### QUESTION 6
# Find the lines along the pens in the ps1-input1.png image
img_clutter = cv2.imread('input/ps1-input2.png')

# convert to grayscale and find edges
img_clutter_grayscale = cv2.cvtColor(img_clutter, cv2.COLOR_RGB2GRAY)
img_clutter_blur = cv2.GaussianBlur(img_clutter_grayscale, (7, 7), 1)
img_clutter_edges = cv2.Canny(img_clutter_blur, 100, 200)
cv2.imshow('Original Image', img_clutter)
cv2.imshow('Grayscale Blurred', img_clutter_blur)
cv2.imshow('Canny Edges', img_clutter_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# build the Hough Accumulator
H, rhos, thetas = hough_lines_acc(img_clutter_edges, theta_resolution=0.75)
indicies, H, H1 = hough_peaks(H, 10, nhood_size=21)
plot_hough_acc(H, H1, plot_title='Hough Peaks Highlighted')

# draw lines
hough_lines_draw(img_clutter, indicies, rhos, thetas)
cv2.imshow('Hough Lines', img_clutter)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save initial attempt to find only the pen edges
cv2.imwrite('output/ps1-6-a-1.png', img_clutter)
