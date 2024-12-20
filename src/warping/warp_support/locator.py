import cv2
import numpy as np
import os
import os.path as path
import dlib
from imutils.face_utils import FaceAligner, rect_to_bb
import argparse
import imutils

# Configuration for the location of face landmark model data
DATA_DIR = os.environ.get(
    'DLIB_DATA_DIR',
    path.join(path.dirname(path.dirname(path.realpath(__file__))), 'data')
)


# Initialize Dlib's face detector
# This uses a pre-trained frontal face detector from Dlib's library
dlib_detector = dlib.get_frontal_face_detector()

# Initialize Dlib's facial landmark predictor
# Replace the path below with the path to your 68-face-landmarks model file
# Ensure the file exists at the specified location
shape_predictor_path = "../inversion/weights/shape_predictor_68_face_landmarks.dat"
dlib_predictor = dlib.shape_predictor(shape_predictor_path)

# Notes:
# - `dlib.get_frontal_face_detector()` provides a fast and accurate face detector.
# - `dlib.shape_predictor()` initializes the predictor for detecting 68 face landmarks.
# - Ensure that the `shape_predictor_68_face_landmarks.dat` file is downloaded and accessible.


def boundary_points(points, width_percent=0.1, height_percent=0.1):
    """
    Generate additional boundary points at the top corners of a bounding rectangle.

    This function calculates two additional points located at the top corners of
    a bounding rectangle surrounding the provided points. The new points can be
    adjusted inward or outward based on the width and height percentages.

    :param points: numpy.ndarray
        An *m* x 2 array of (x, y) points representing the key points of an object.

    :param width_percent: float, optional (default=0.1)
        Percentage of the width used to taper the points inward or outward.
        - Values should be in the range [-1, 1].
        - Positive values move the points inward, negative values move them outward.

    :param height_percent: float, optional (default=0.1)
        Percentage of the height used to taper the points upward or downward.
        - Values should be in the range [-1, 1].
        - Positive values move the points downward, negative values move them upward.

    :returns: list
        A list of two new points represented as [[x1, y1], [x2, y2]].
        These points are located at the adjusted top corners of the bounding rectangle.
    """
    # Calculate the bounding rectangle for the input points
    x, y, w, h = cv2.boundingRect(np.array([points], np.int32))

    # Calculate the offset distances for width and height adjustments
    spacerw = int(w * width_percent)
    spacerh = int(h * height_percent)

    # Generate the two new boundary points at the adjusted top corners
    return [[x + spacerw, y + spacerh],
            [x + w - spacerw, y + spacerh]]

def face_points(img, size, add_boundary_points=True):
    """
    Wrapper function to detect face points in an image using Dlib.

    This function delegates the task of face point detection to `face_points_dlib`.
    Optionally, boundary points can be added to the detected points.

    :param img: numpy.ndarray
        Input image in which face points need to be detected.

    :param size: tuple (height, width)
        Desired output size for the detected face points.

    :param add_boundary_points: bool, optional (default=True)
        If True, additional boundary points are added to the detected face points.

    :returns: numpy.ndarray
        Array of face points with or without additional boundary points, depending on the value of `add_boundary_points`.
    """
    return face_points_dlib(img, size, add_boundary_points)

def face_points_dlib(img, size, add_boundary_points=True):
    """
    Locate 68 facial landmarks in an image using Dlib's shape predictor.

    This function uses the Dlib library to detect facial landmarks and optionally
    adds additional boundary points to enhance the detected points. Ensure the
    required shape predictor model file is downloaded and available at the specified path.

    :param img: numpy.ndarray
        Input image array in BGR format.

    :param size: tuple (height, width)
        Desired output size to add corner points at the boundaries.

    :param add_boundary_points: bool, optional (default=True)
        If True, additional boundary points are added beyond the detected landmarks.

    :returns: numpy.ndarray
        Array of (x, y) coordinates of the detected facial landmarks and boundary points.
        Returns an empty array if no face is found.

    :raises: Exception
        Prints the exception message and returns an empty array in case of an error.
    """

    try:
        points = []

        # Convert the image to RGB as required by Dlib
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        rects = dlib_detector(rgbimg, 1)

        if rects and len(rects) > 0:
            # Use the first detected face
            shapes = dlib_predictor(rgbimg, rects[0])
            points = np.array([(shapes.part(i).x, shapes.part(i).y) for i in range(68)], np.int32)

            if add_boundary_points:
                # Add additional boundary points beyond the 68 detected points
                points = np.vstack([
                    points,
                    boundary_points(points, 0.1, -0.03),
                    boundary_points(points, 0.13, -0.05),
                    boundary_points(points, 0.15, -0.08),
                    boundary_points(points, 0.33, -0.12)
                ])

        # Add corner points of the image based on the provided size
        points = np.vstack([
            points,
            [[1, 1], [size[1] - 2, 1], [1, size[0] - 2], [size[1] - 2, size[0] - 2]]
        ])

        return points
    except Exception as e:
        # Print the exception and return an empty array
        print(e)
        return []