import cv2
import numpy as np
import scipy.sparse


def mask_from_points(size, points, iter=1, radius=40):
    """
    Create a mask of the supplied size from the supplied points.

    :param size: tuple
        Dimensions of the output mask as (height, width).

    :param points: numpy.ndarray
        Array of (x, y) points to define the convex hull for the mask.

    :param iter: int, optional (default=1)
        Number of iterations for the erosion operation on the mask.

    :param radius: int, optional (default=40)
        Radius of the kernel used for erosion.

    :returns: numpy.ndarray
        Binary mask with values 0 and 255, where 255 indicates the convex hull region.
    """
    kernel = np.ones((radius, radius), np.uint8)
    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel, iter)
    return mask

def overlay_image(foreground_image, mask, background_image):
    """
    Overlay the foreground image onto the background image based on a mask.

    This function uses a binary mask to blend the foreground image onto the background image.
    Only the regions where the mask is non-zero are replaced with the corresponding pixels
    from the foreground image.

    :param foreground_image: numpy.ndarray
        The image to overlay onto the background. Should have the same dimensions as the background image.

    :param mask: numpy.ndarray
        A binary mask with values in the range [0, 255]. The mask determines which parts of the
        foreground image will overlay onto the background image. Non-zero values indicate
        the regions to overlay.

    :param background_image: numpy.ndarray
        The image onto which the foreground image will be overlaid. Should have the same
        dimensions as the foreground image.

    :returns: numpy.ndarray
        The resulting image after overlaying the foreground onto the background.
    """
    # Determine the regions where the mask is non-zero
    foreground_pixels = mask > 0

    # Replace the corresponding pixels in the background image with the foreground image
    background_image[..., :3][foreground_pixels] = foreground_image[..., :3][foreground_pixels]

    return background_image

def apply_mask(img, mask):
    """
    Apply a binary mask to the supplied image.

    This function multiplies each channel of the input image by the normalized values
    of the mask (scaled between 0 and 1) to produce a masked image.

    :param img: numpy.ndarray
        Input image with a maximum of 3 channels (e.g., RGB or grayscale).

    :param mask: numpy.ndarray
        A binary mask with values in the range [0, 255]. Non-zero values indicate the
        regions of the image to retain.

    :returns: numpy.ndarray
        The resulting image with the mask applied.
    """
    # Create a copy of the input image to preserve the original
    masked_img = np.copy(img)

    # Determine the number of channels to process
    num_channels = min(3, img.shape[-1]) if img.ndim == 3 else 1

    # Apply the mask to each channel
    for c in range(num_channels):
        masked_img[..., c] = img[..., c] * (mask / 255)

    return masked_img

def correct_colours(im1, im2, landmarks1):
    """
    Adjust the colors of one image to match another based on facial landmarks.

    This function performs color correction by aligning the color distribution of `im1`
    to match that of `im2`, using Gaussian blur and a ratio-based correction. The eyes
    are used as reference points for calculating the blur amount.

    :param im1: numpy.ndarray
        The source image whose colors need to be corrected.

    :param im2: numpy.ndarray
        The target image whose color distribution serves as the reference.

    :param landmarks1: numpy.ndarray
        Facial landmarks for the source image (`im1`). Used to calculate the blur
        amount based on the distance between the left and right eyes.

    :returns: numpy.ndarray
        The color-corrected version of the source image (`im1`).
    """
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(37, 43))
    RIGHT_EYE_POINTS = list(range(43, 49))

    # Calculate the blur amount based on the distance between the eyes
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)

    # Ensure the blur amount is odd
    if blur_amount % 2 == 0:
        blur_amount += 1

    # Apply Gaussian blur to both images
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors by adjusting low values in the blurred target image
    im2_blur = im2_blur.astype(int)
    im2_blur += 128 * (im2_blur <= 1)

    # Perform the color correction
    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
