
import os
import numpy as np
import cv2
from warping.warp_support import plotter, aligner, warper, blender, plotter, videoer

def verify_args(args):
  if args['--images'] is None:
    valid = os.path.isfile(args['--src']) & os.path.isfile(args['--dest'])
    if not valid:
      print('--src=%s or --dest=%s file does not exist. Double check the supplied paths' % (
        args['--src'], args['--dest']))
      exit(1)
  else:
    valid = os.path.isdir(args['--images'])
    if not valid:
      print('--images=%s is not a valid directory' % args['--images'])
      exit(1)

def load_image_points(path, size):
  img = cv2.imread(path)
  if "MyFace" in path:
      img = locator.align(img,size)
  points = locator.face_points(img,size)

  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    return aligner.resize_align(img, points, size)

def load_valid_image_points(imgpaths, size):
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      print(path)
      yield (img, points)

def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)





def get_pasted_masks(file1, file2, output_dir, size=(1024, 1024)):
    """
    Generate and save images with pasted masks using seamless cloning.

    This function loads two input images, processes them to generate masks, warps the images,
    and applies seamless cloning to produce the final output images with pasted masks.

    :param file1: str
        Path to the first input image.

    :param file2: str
        Path to the second input image.

    :param output_dir: str
        Directory where the output images will be saved.

    :param size: tuple, optional (default=(1024, 1024))
        The desired size (width, height) for processing the images.

    :returns: None
        Saves the processed images with pasted masks to the specified output directory.
    """
    # Load the input images
    source_face = cv2.imread(file1)
    dest_face = cv2.imread(file2)
    H, W, D = source_face.shape

    # Extract filenames without extensions
    first = file1.split("/")[-1].split(".")[0]
    second = file2.split("/")[-1].split(".")[0]

    try:
        # Morph the images to generate masks
        left, right = morpher(
            None,
            list_imgpaths(None, file1, file2),
            source_face,
            dest_face,
            int(H),
            int(W),
            0,
            0,
            "nothing",
            None,
            None,
            "average",
            get_masks=True,
        )

        # Process the generated masks if valid
        if isinstance(left, np.ndarray):
            # Load image points for both input images
            original_1, original1_points = load_image_points(file1, size)
            original_2, original2_points = load_image_points(file2, size)

            # Compute canonical points using weighted average
            points = locator.weighted_average_points(original2_points, original1_points, 0.5)

            # Process the first image with its mask
            original_1_warped = warper.warp_image(original_1, original1_points, points, size)
            temp_points = points[:-4]
            mask = blender.mask_from_points(size, temp_points)
            r = cv2.boundingRect(mask)
            center = (r[0] + int(r[2] // 2), r[1] + int(r[3] // 2))

            if isinstance(original_1_warped, (list, np.ndarray)) and isinstance(left, (list, np.ndarray)):
                new_image_1 = cv2.seamlessClone(left, original_1_warped, mask, center, cv2.NORMAL_CLONE)
                cv2.imwrite(os.path.join(output_dir, f"{first}_{second}.png"), new_image_1)

            # Process the second image with its mask
            original_2_warped = warper.warp_image(original_2, original2_points, points, size)
            mask = blender.mask_from_points(size, temp_points)
            r = cv2.boundingRect(mask)
            center = (r[0] + int(r[2] // 2), r[1] + int(r[3] // 2))

            if isinstance(original_2_warped, (list, np.ndarray)) and isinstance(right, (list, np.ndarray)):
                new_image_2 = cv2.seamlessClone(right, original_2_warped, mask, center, cv2.NORMAL_CLONE)
                cv2.imwrite(os.path.join(output_dir, f"{second}_{first}.png"), new_image_2)

            print("Saved: ", os.path.join(output_dir, f"{first}_{second}.png"))
            print("Saved: ", os.path.join(output_dir, f"{second}_{first}.png"))

    except Exception as e:
        print("Error Warping", file1, file2, ":", str(e))

def get_masks(file1, file2, output_dir):
    """
    Generate and save face masks by morphing two input images.

    Args:
        file1 (str): Path to the first input image.
        file2 (str): Path to the second input image.
        output_dir (str): Directory to save the output masks.

    Returns:
        None
    """
    # Load the input images
    source_face = cv2.imread(file1)
    dest_face = cv2.imread(file2)

    # Get dimensions of the source image
    W, H, D = source_face.shape

    try:
        # Perform morphing to generate masks
        left, right = morpher(
            None,  # Placeholder for additional arguments (if any)
            list_imgpaths(None, file1, file2),  # Generate list of image paths
            source_face,  # Source face image
            dest_face,    # Destination face image
            int(H),       # Height of the image
            int(W),       # Width of the image
            0, 0,         # Placeholder values for additional arguments (if any)
            "nothing", None, None, "average", get_masks=True
        )

        # Check if the returned masks are valid arrays
        if isinstance(left, np.ndarray):
            # Extract filenames (without extensions) for saving
            first = file1.split("/")[-1].split(".")[0]
            second = file2.split("/")[-1].split(".")[0]

            # Save the generated masks
            cv2.imwrite(os.path.join(output_dir, f"{first}_{second}.png"), left)
            cv2.imwrite(os.path.join(output_dir, f"{second}_{first}.png"), right)

            # Print confirmation messages
            print("Saved:", os.path.join(output_dir, f"{first}_{second}.png"))
            print("Saved:", os.path.join(output_dir, f"{second}_{first}.png"))

    except Exception as e:
        # Handle exceptions and print error message
        print("Error Warping", file1, file2)
        print("Exception:", e)


def paste_images(o1, o2, m, output_1, output_2, size=(1024, 1024)):
    """
    Paste masks onto background images using seamless cloning.

    Args:
        o1 (str): Path to the first background image.
        o2 (str): Path to the second background image.
        m (str): Path to the mask image.
        output_1 (str): Path to save the first output image.
        output_2 (str): Path to save the second output image.
        size (tuple): Target size for all images (default: (1024, 1024)).

    Returns:
        None
    """
    # Load images and corresponding points
    morph, morph_points = load_image_points(m, size)
    original_1, original1_points = load_image_points(o1, size)
    original_2, original2_points = load_image_points(o2, size)

    # Ensure all images and points are loaded successfully
    if all(
        x is not None for x in [original_1, original_2, original1_points, original2_points, morph, morph_points]
    ):
        # Compute canonical points by averaging points from both images
        points = locator.weighted_average_points(original2_points, original1_points, 0.5)

        # Warp the first original image
        original_1_warped = warper.warp_image(original_1, original1_points, points, size)

        # Extract mask and calculate the center of the region for cloning
        temp_points = points[:-4]
        mask = blender.mask_from_points(size, temp_points)
        r = cv2.boundingRect(mask)
        center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

        # Perform seamless cloning for the first image
        if isinstance(original_1_warped, (list, np.ndarray)) and isinstance(morph, (list, np.ndarray)):
            new_image_1 = cv2.seamlessClone(morph, original_1_warped, mask, center, cv2.NORMAL_CLONE)
            cv2.imwrite(output_1, new_image_1)

        # Warp the second original image
        original_2_warped = warper.warp_image(original_2, original2_points, points, size)

        # Reuse mask and center for the second image
        mask = blender.mask_from_points(size, temp_points)
        r = cv2.boundingRect(mask)
        center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

        # Perform seamless cloning for the second image
        if isinstance(original_2_warped, (list, np.ndarray)) and isinstance(morph, (list, np.ndarray)):
            new_image_2 = cv2.seamlessClone(morph, original_2_warped, mask, center, cv2.NORMAL_CLONE)
            cv2.imwrite(output_2, new_image_2)

        # Print confirmation of output creation
        print("Made:", output_1, output_2)
