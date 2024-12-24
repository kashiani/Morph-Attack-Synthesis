import cv2
import numpy as np
import scipy.sparse


def mask_from_points(size, points, iter=1, radius=40):
    """
    Create a mask of the supplied size from the supplied points.

    :param size: tuple
        Dimensions of the output mask as (height, width).

    :param points: numpy.ndarray
        Array of (x, y) points to define the convex hull for the mask.

    :param iter: int, optional (default=1)
        Number of iterations for the erosion operation on the mask.

    :param radius: int, optional (default=40)
        Radius of the kernel used for erosion.

    :returns: numpy.ndarray
        Binary mask with values 0 and 255, where 255 indicates the convex hull region.
    """
    kernel = np.ones((radius, radius), np.uint8)
    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel, iter)
    return mask