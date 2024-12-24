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

def overlay_image(foreground_image, mask, background_image):
    """
    Overlay the foreground image onto the background image based on a mask.

    This function uses a binary mask to blend the foreground image onto the background image.
    Only the regions where the mask is non-zero are replaced with the corresponding pixels
    from the foreground image.

    :param foreground_image: numpy.ndarray
        The image to overlay onto the background. Should have the same dimensions as the background image.

    :param mask: numpy.ndarray
        A binary mask with values in the range [0, 255]. The mask determines which parts of the
        foreground image will overlay onto the background image. Non-zero values indicate
        the regions to overlay.

    :param background_image: numpy.ndarray
        The image onto which the foreground image will be overlaid. Should have the same
        dimensions as the foreground image.

    :returns: numpy.ndarray
        The resulting image after overlaying the foreground onto the background.
    """
    # Determine the regions where the mask is non-zero
    foreground_pixels = mask > 0

    # Replace the corresponding pixels in the background image with the foreground image
    background_image[..., :3][foreground_pixels] = foreground_image[..., :3][foreground_pixels]

    return background_image
