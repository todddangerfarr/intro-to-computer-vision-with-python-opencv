###############################################################################
#       PERSPECTIVE IMAGING 5.2: Stereo Correspondence - Find Best Match      #
#                                By: Todd Farr                                #
###############################################################################

# Using a window-based stereo matching method to simply calculate a rough
# disparity between strips in corresponding stereo images.

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load images
left = cv2.imread('images/flowers-left.png')
right = cv2.imread('images/flowers-right.png')

# convert to grayscale
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# show grayscale images
cv2.imshow('Left', left_gray)
cv2.imshow('Right', right_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# define the image patch
patch_loc = [100, 120]
patch_size = [100, 100]

# extract the patch from the left image
patch_left = left_gray[patch_loc[0]:(patch_loc[0] + patch_size[0]),
                       patch_loc[1]:(patch_loc[1] + patch_size[1])]

# extract strip from the rigth image
strip_right = right_gray[patch_loc[0]:(patch_loc[0] + patch_size[0]), :]

# show left patch and right strip
cv2.imshow('Image Patch Left', patch_left)
cv2.imshow('Image Strip Right', strip_right)
cv2.waitKey(0)
cv2.destroyAllWindows()


################################### FIND BEST X USING SUM OF SQUARED DIFFERENCES
# create a function to find the best x match location in the right strip
def best_x(patch, strip):
    """ Return the best x location for the patch in the strip using the sum of
    squared differences.

    Params:
    patch: The input image patch to be compared
    strip: The input strip image in which the patch is compared against

    Returns:
    best_x: The best x fit location of the patch in the strip

    """
    min_diff = float('inf')
    best_x = 0
    for x in range(0, strip.shape[1] - patch.shape[1] + 1):
        extracted_patch = strip[:, x:(x + patch.shape[1])]
        ssd = np.sum((patch - extracted_patch)**2)
        if ssd < min_diff:
            best_x = x
            min_diff = ssd

    return best_x

# get best_x and corresponding patch
x = best_x(patch_left, strip_right)
patch_right = strip_right[:, x:(x + patch_left.shape[1])]

# plot patch left, strip right and patch right using matplotlib
fig = plt.figure(figsize=(8, 6))
fig.canvas.set_window_title('Best X Patch Detection')
plt.subplot(311), plt.imshow(patch_left, 'gray')
plt.title('Original Patch from the Left Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(strip_right, 'gray')
plt.title('Original Strip from the Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(313), plt.imshow(patch_right, 'gray')
plt.title('Extracted Patch from the Right Image'), plt.xticks([]), plt.yticks([])

plt.show()


############################################# FIND DISPARITY VECTOR FOR 2 STRIPS
# define strip row (y) and the square block size (b)
y = 75
b = 100

# extract strip from the images
left_strip = left_gray[y:(y + b), :]
right_strip = right_gray[y:(y + b), :]

# show strip images
cv2.imshow('Left Disparity Image Strip', left_strip)
cv2.imshow('Right Disparity Image Strip', right_strip)
cv2.waitKey(0)
cv2.destroyAllWindows()

# define function to find disparity
def match_strips(left_strip, right_strip, block_size):
    ''' A function for finding the pixel disparity vector between each block in
        right and left strips with the given block size. '''
    num_blocks = left_strip.shape[1] / block_size # python 2 floors integer div
    # create an empty disparity vector
    disparity = np.zeros(num_blocks)
    # how many times does the block fit in the strip
    for i in range(num_blocks):
        x_left = i*block_size
        patch_left = left_strip[:, x_left:(x_left + block_size)] # get patch
        x_right = best_x(patch_left, right_strip)
        disparity[i] = x_left - x_right

    return disparity

# call the match strips function
disparity = match_strips(left_strip, right_strip, b)
print disparity

# plot strips and disparity
fig = plt.figure(figsize=(8, 6))
fig.canvas.set_window_title('Stereo Disparity')
plt.subplot(311), plt.imshow(left_strip, 'gray')
plt.title('Strip from the Left Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(right_strip, 'gray')
plt.title('Strip from the Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(313), plt.plot(disparity)
plt.title('Disparity Between the Two Images')

plt.show()
