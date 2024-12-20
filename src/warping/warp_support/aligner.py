"""
Align face and image sizes
"""
import cv2
import numpy as np

def positive_cap(num):
  """ Cap a number to ensure positivity

  :param num: positive or negative number
  :returns: (overflow, capped_number)
  """
  if num < 0:
    return 0, abs(num)
  else:
    return num, 0


def roi_coordinates(rect, size, scale):
  """ Align the rectangle into the center and return the top-left coordinates
  within the new size. If rect is smaller, we add borders.

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :param scale: scaling factor of the rectangle to be resized
  :returns: 4 numbers. Top-left coordinates of the aligned ROI.
    (x, y, border_x, border_y). All values are > 0.
  """
  rectx, recty, rectw, recth = rect
  new_height, new_width = size
  mid_x = int((rectx + rectw/2) * scale)
  mid_y = int((recty + recth/2) * scale)
  roi_x = mid_x - int(new_width/2)
  roi_y = mid_y - int(new_height/2)

  roi_x, border_x = positive_cap(roi_x)
  roi_y, border_y = positive_cap(roi_y)
  return roi_x, roi_y, border_x, border_y

def scaling_factor(rect, size):
    """
    Calculate the scaling factor for resizing an image to new dimensions.

    The function determines the scaling factor based on the bounding rectangle
    dimensions of a face and the desired size. It ensures the new dimensions
    maintain a consistent aspect ratio with a reduction factor of 80%.

    :param rect: tuple (x, y, w, h)
        Bounding rectangle of the face, where:
        - x, y: Coordinates of the top-left corner (ignored in the calculation).
        - w: Width of the bounding rectangle.
        - h: Height of the bounding rectangle.

    :param size: tuple (width, height)
        Desired dimensions of the output image:
        - width: Target width.
        - height: Target height.

    :returns: float
        Scaling factor as a floating-point number.
    """
    # Unpack desired dimensions
    new_height, new_width = size

    # Extract the width and height from the bounding rectangle
    rect_h, rect_w = rect[2:]

    # Calculate the height and width scaling ratios
    height_ratio = rect_h / new_height
    width_ratio = rect_w / new_width

    # Initialize scaling factor
    scale = 1

    # Determine scaling based on the dominant ratio
    if height_ratio > width_ratio:
        # Height is the limiting dimension
        new_rect_h = 0.8 * new_height  # Reduced target height by 80%
        scale = new_rect_h / rect_h
    else:
        # Width is the limiting dimension
        new_rect_w = 0.8 * new_width  # Reduced target width by 80%
        scale = new_rect_w / rect_w

    return 1


def resize_image(img, scale):
    """
    Resize an image using the specified scaling factor.

    This function adjusts the dimensions of the input image proportionally
    by multiplying its width and height by the given scaling factor.

    :param img: numpy.ndarray
        Input image to be resized. It should be a valid image array with dimensions
        (height, width, channels) or (height, width).

    :param scale: float
        Scaling factor to resize the image. Values greater than 1 enlarge the image,
        while values between 0 and 1 shrink it.

    :returns: numpy.ndarray
        The resized image with adjusted dimensions based on the scaling factor.
    """
    # Retrieve the current dimensions of the image
    cur_height, cur_width = img.shape[:2]

    # Calculate new dimensions based on the scaling factor
    new_scaled_height = int(scale * cur_height)
    new_scaled_width = int(scale * cur_width)

    # Resize the image using OpenCV's resize function
    return cv2.resize(img, (new_scaled_width, new_scaled_height))

def resize_align(img, points, size):
    """
    Resize an image and associated points, align the face to the center,
    and crop it to the desired size.

    This function resizes the input image based on a bounding rectangle,
    aligns the face to the center of the output, and adjusts the points
    accordingly to maintain alignment with the transformed image.

    :param img: numpy.ndarray
        Input image to be resized. Expected to be a valid image array.

    :param points: numpy.ndarray
        Array of shape (*m* x 2), representing *m* facial landmarks or key points
        with (x, y) coordinates.

    :param size: tuple (height, width)
        Desired output size of the image after resizing and cropping.

    :returns: tuple (numpy.ndarray, numpy.ndarray)
        - Resized and cropped image as a numpy array with the specified dimensions.
        - Adjusted points array of the same shape as the input points.
    """

    # Unpack desired dimensions
    new_height, new_width = size

    # Compute bounding rectangle for the points
    rect = cv2.boundingRect(np.array([points], np.int32))

    # Calculate scaling factor based on the bounding rectangle and desired size
    scale = scaling_factor(rect, size)

    # Resize the image using the calculated scaling factor
    img = resize_image(img, scale)

    # Align the bounding rectangle to the center of the image
    cur_height, cur_width = img.shape[:2]
    roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
    roi_h = np.min([new_height - border_y, cur_height - roi_y])
    roi_w = np.min([new_width - border_x, cur_width - roi_x])

    # Crop the image to the desired size
    crop = np.zeros((new_height, new_width, 3), img.dtype)
    crop[border_y:border_y + roi_h, border_x:border_x + roi_w] = (
        img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    )

    # Adjust the facial points to match the scaled and cropped image
    points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
    points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

    return crop, points


