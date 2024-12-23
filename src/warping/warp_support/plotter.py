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

def check_do_plot(func):
    """
    Decorator to execute a plotting function only if `do_plot` is enabled.

    :param func: function
        The function to conditionally execute.

    :returns: function
        Wrapped function that checks `do_plot` before execution.
    """
    def inner(self, *args, **kwargs):
        if self.do_plot:
            func(self, *args, **kwargs)
    return inner

def check_do_save(func):
    """
    Decorator to execute a save function only if `do_save` is enabled.

    :param func: function
        The function to conditionally execute.

    :returns: function
        Wrapped function that checks `do_save` before execution.
    """
    def inner(self, *args, **kwargs):
        if self.do_save:
            func(self, *args, **kwargs)
    return inner