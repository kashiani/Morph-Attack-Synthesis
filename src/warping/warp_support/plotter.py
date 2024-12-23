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

class Plotter:
    """
    A class for plotting and saving images with additional utility functions.

    :param plot: bool, optional (default=True)
        Enable or disable plotting functionality.

    :param rows: int, optional (default=0)
        Number of rows in the plot layout. Calculated automatically if zero.

    :param cols: int, optional (default=0)
        Number of columns in the plot layout. Calculated automatically if zero.

    :param num_images: int, optional (default=0)
        Total number of images to plot. Used for automatic layout calculation.

    :param out_folder: str, optional (default=None)
        Folder path for saving images.

    :param out_filename: str, optional (default=None)
        Default filename for saving images.
    """
    def __init__(self, plot=True, rows=0, cols=0, num_images=0, out_folder=None, out_filename=None):
        self.save_counter = 1
        self.plot_counter = 1
        self.do_plot = plot
        self.do_save = out_filename is not None
        self.out_filename = out_filename
        self.set_filepath(out_folder)
        if (rows + cols) == 0 and num_images > 0:
            # Auto-calculate the number of rows and cols for the figure
            self.rows = int(np.ceil(np.sqrt(num_images / 2.0)))
            self.cols = int(np.ceil(num_images / self.rows))
        else:
            self.rows = rows
            self.cols = cols

    def set_filepath(self, folder):
        """
        Set the filepath for saving images.

        :param folder: str
            Directory where images will be saved.
        """
        if folder is None:
            self.filepath = None
            return

