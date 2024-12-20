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


