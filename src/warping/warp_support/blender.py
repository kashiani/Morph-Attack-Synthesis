import cv2
import numpy as np
import scipy.sparse

import pyamg


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

def weighted_average(img1, img2, percent=0.5):
    """
    Compute a weighted average of two images.

    This function blends two images together by applying a weighted average based on the specified percentage.

    :param img1: numpy.ndarray
        The first input image.

    :param img2: numpy.ndarray
        The second input image. Must have the same dimensions as `img1`.

    :param percent: float, optional (default=0.5)
        The weight for `img1`. The weight for `img2` is computed as `1 - percent`.
        - A value of 0 returns `img2` entirely.
        - A value of 1 returns `img1` entirely.

    :returns: numpy.ndarray
        The resulting blended image.
    """
    if percent <= 0:
        return img2
    elif percent >= 1:
        return img1
    else:
        return cv2.addWeighted(img1, percent, img2, 1 - percent, 0)


def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
    """
    Blend two images using alpha feathering.

    This function uses a blurred mask to smoothly blend the source image into the destination image.

    :param src_img: numpy.ndarray
        The source image to blend.

    :param dest_img: numpy.ndarray
        The destination image onto which the source image is blended.

    :param img_mask: numpy.ndarray
        A binary mask defining the region of interest for blending. Non-zero values indicate the region to blend.

    :param blur_radius: int, optional (default=15)
        The radius of the blur applied to the mask for smooth transitions.

    :returns: numpy.ndarray
        The resulting image after blending.
    """
    # Apply blur to the mask for feathering
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    # Initialize the result image
    result_img = np.empty(src_img.shape, np.uint8)

    # Blend each channel separately
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1 - mask)

    return result_img



def poisson_blend(img_source, dest_img, img_mask, offset=(0, 0)):
    """
    Blend two images seamlessly using Poisson blending.

    This function adjusts the source image to blend naturally with the destination image,
    using the provided mask and offset to determine the region to blend.

    :param img_source: numpy.ndarray
        The source image to blend into the destination image.

    :param dest_img: numpy.ndarray
        The destination image onto which the source image is blended.

    :param img_mask: numpy.ndarray
        A binary mask defining the region of the source image to blend. Non-zero values indicate the region to blend.

    :param offset: tuple, optional (default=(0, 0))
        The (x, y) offset to position the source image relative to the destination image.

    :returns: numpy.ndarray
        The resulting image after Poisson blending.
    """
    img_target = np.copy(dest_img)

    return img_target
