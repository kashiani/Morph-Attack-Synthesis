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
