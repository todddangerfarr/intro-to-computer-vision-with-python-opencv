###############################################################################
#                 PERSPECTIVE IMAGING 5.0: Project a Point                    #
#                                By: Todd Farr                                #
###############################################################################

# Project a point from 3D to 2D using a matrix operation

# Given: Point p in 3-space [x y z], and focal length f
# Return: Location of projected point on 2D image plane [u v]

# imports
import numpy as np

def project_point(p, f):
    ''' Projects a point in 3-space onto a 2D image plane using a focal lenght
        of f. '''
    x, y, z = p
    projection_matrix = np.matrix([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0]])
    homogenous_vector = np.matrix([[x], [y], [z], [1]])
    p_proj = projection_matrix * homogenous_vector
    return ((p_proj.item(0) / p_proj.item(2)),
            (p_proj.item(1) / p_proj.item(2)))

# check values
point = (200, 100, 120)
point2 = (200, 100, 50) # any point that has z of the focal length returns x, y
point3 = (200, 100, 100) # points at twice the focal length should be half size
focal_length = 50

# call function
print project_point(point, focal_length)
print project_point(point2, focal_length)
print project_point(point3, focal_length)
