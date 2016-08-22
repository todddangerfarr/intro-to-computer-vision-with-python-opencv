# PROBLEM SET 1

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

##################################################################### QUESTION 1
# Load the input grayscale image (input/ps1-input0.png) as img and
# generate an edge image - which is a binary image with white pixels (1) on the
# edges and black pixels (0) elsewhere.  Use one operator of your choosing - for
# this image it probably won't matter much. If your edge operator uses
# parameters (like 'canny') play with those until you get the edges you would
# expect to see.
img = cv2.imread('input/ps1-input0.png')
img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_edges = cv2.Canny(img_grayscale, 100, 200)
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
    x_idxs, y_idxs = np.nonzero(img) # find all edge pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


# Step 2:  Find the num highest peaks from Hough Line Accumulator
def hough_peaks(H, num_peaks):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to local maxima. '''
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


def hough_lines_draw(img, indicies, rhos, thetas):
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
H, rhos, thetas = hough_lines_acc(img_edges)

# plot the Hough Line Accumulator
fig = plt.figure(figsize=(6, 10))
fig.canvas.set_window_title('Hough Accumulator Plot')
plt.imshow(H, cmap='gray')
plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
plt.tight_layout()
plt.show()

indicies = hough_peaks(H, 6)
print indicies

hough_lines_draw(img, indicies, rhos, thetas)
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
