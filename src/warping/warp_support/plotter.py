import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2

def bgr2rgb(img):
    """
    Convert an image from BGR to RGB format.

    :param img: numpy.ndarray
        Input image in BGR format.

    :returns: numpy.ndarray
        Image converted to RGB format.
    """
    rgb = np.copy(img)
    rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
    return rgb

