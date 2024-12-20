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
