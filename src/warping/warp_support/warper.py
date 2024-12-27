import numpy as np
import scipy.spatial as spatial


import numpy as np

def bilinear_interpolate(img, coords):
    """
    Perform bilinear interpolation on a given image at specified coordinates.

    Bilinear interpolation estimates the intensity of a pixel based on the intensities
    of its four nearest neighbors. This implementation handles up to 3-channel images.

    :param img: numpy.ndarray
        Input image with a maximum of 3 channels.

    :param coords: numpy.ndarray
        A 2 x _m_ array of coordinates where:
        - First row contains x-coordinates.
        - Second row contains y-coordinates.

    :returns: numpy.ndarray
        An array of interpolated pixel values with the same number of channels as the input image
        and shape matching the input coordinates.

    :raises: ValueError
        If the coordinates are out of bounds of the image dimensions.
    """
    # Convert coordinates to integer indices for the nearest neighbors
    int_coords = np.int32(coords)
    x0, y0 = int_coords

    # Calculate the fractional part of the coordinates
    dx, dy = coords - int_coords

    # Check if coordinates are within image bounds
    if (x0 < 0).any() or (x0 >= img.shape[1] - 1).any() or (y0 < 0).any() or (y0 >= img.shape[0] - 1).any():
        raise ValueError("Coordinates are out of image bounds.")

    # Extract intensity values from the 4 neighboring pixels
    q11 = img[y0, x0]       # Top-left neighbor
    q21 = img[y0, x0 + 1]   # Top-right neighbor
    q12 = img[y0 + 1, x0]   # Bottom-left neighbor
    q22 = img[y0 + 1, x0 + 1]  # Bottom-right neighbor

    # Interpolate horizontally between the top neighbors and the bottom neighbors
    btm = q21.T * dx + q11.T * (1 - dx)  # Bottom interpolation
    top = q22.T * dx + q12.T * (1 - dx)  # Top interpolation

    # Interpolate vertically between the top and bottom results
    inter_pixel = top * dy + btm * (1 - dy)

    # Transpose the result back to original orientation
    return inter_pixel.T