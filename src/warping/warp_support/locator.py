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