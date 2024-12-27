import numpy as np
import scipy.spatial as spatial


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


def grid_coordinates(points):
    """
    Generate a grid of x, y coordinates within the region of interest (ROI) defined by the input points.

    This function computes a rectangular grid of coordinates that spans the minimum and maximum x, y values
    of the supplied points.

    :param points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents a point's (x, y) coordinates.

    :returns: numpy.ndarray
        An array of shape (m, 2) containing (x, y) coordinates for the grid, where:
        - `m` is the total number of grid points.
    """
    # Determine the bounding box of the points
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1  # Include the endpoint
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1  # Include the endpoint

    # Generate grid coordinates within the bounding box
    grid = np.asarray(
        [(x, y) for y in range(ymin, ymax) for x in range(xmin, xmax)],
        dtype=np.uint32
    )

    return grid

import numpy as np

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the source image to the destination image within the ROI defined by destination points.

    This function applies affine transformations to warp each triangle of the source image into its corresponding
    triangle in the destination image. The warping is restricted to the ROI determined by the destination points.

    :param src_img: numpy.ndarray
        The source image to be warped.

    :param result_img: numpy.ndarray
        The destination image where the warped triangles will be placed. Should have the same shape as the source image.

    :param tri_affines: list of numpy.ndarray
        A list of 2x3 affine transformation matrices, one for each triangle.

    :param dst_points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents the (x, y) coordinates of the destination points.

    :param delaunay: scipy.spatial.Delaunay
        A Delaunay triangulation object containing the simplices (triangles) and their connectivity.

    :returns: None
        The function modifies `result_img` in place by applying the warped triangles.
    """
    # Generate grid coordinates within the ROI of the destination points
    roi_coords = grid_coordinates(dst_points)

    # Map coordinates to their respective triangles; -1 if outside any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    # Iterate over each triangle in the Delaunay triangulation
    for simplex_index in range(len(delaunay.simplices)):
        # Get coordinates within the current triangle
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)

        # If there are no coordinates in this triangle, skip
        if num_coords == 0:
            continue

        # Transform coordinates using the affine matrix
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))

        # Extract x, y coordinates for destination
        x, y = coords.T

        # Apply bilinear interpolation from the source image to the result image
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None

def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate affine transformation matrices for each triangle defined by vertices.

    This function computes the 2x3 affine transformation matrix for each triangle,
    mapping points from the destination image to the source image.

    :param vertices: numpy.ndarray
        An array of triplet indices, where each triplet represents the corners of a triangle.

    :param src_points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents [x, y] coordinates of landmarks in the source image.

    :param dest_points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents [x, y] coordinates of landmarks in the destination image.

    :yields: numpy.ndarray
        A 2x3 affine transformation matrix for each triangle, mapping destination points to source points.
    """
    # Constant row of ones to append for affine matrix computation
    ones = [1, 1, 1]

    # Iterate through each triangle defined by the vertex indices
    for tri_indices in vertices:
        # Extract the triangle's corner points from the source and destination landmarks
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))

        # Compute the affine transformation matrix
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]

        # Yield the 2x3 affine matrix for the current triangle
        yield mat


def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    """
    Warp an image from source landmarks to destination landmarks.

    This function maps the source image to the destination shape by warping it based on
    the corresponding source and destination points. Delaunay triangulation is used to
    define the triangular regions for warping.

    :param src_img: numpy.ndarray
        The source image to be warped. Must have at least 3 channels (RGB).

    :param src_points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents [x, y] coordinates of landmarks in the source image.

    :param dest_points: numpy.ndarray
        A 2D array of shape (n, 2), where each row represents [x, y] coordinates of landmarks in the destination image.

    :param dest_shape: tuple
        The shape (rows, cols) of the destination image.

    :param dtype: numpy.dtype, optional (default=np.uint8)
        The data type of the output image.

    :returns: numpy.ndarray
        The warped image with the specified destination shape and 3 channels.
    """
    # Ensure the resultant image will not have an alpha channel
    num_chans = 3

    # Remove alpha channel from source image if present
    src_img = src_img[:, :, :3]

    # Initialize the result image with zeros
    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows, cols, num_chans), dtype)

    # Perform Delaunay triangulation on the destination points
    delaunay = spatial.Delaunay(dest_points)

    # Compute affine transformation matrices for all triangles
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dest_points)))

    # Warp the source image onto the result image using the computed affine matrices
    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

    return result_img
