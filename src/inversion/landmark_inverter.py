"""
Landmark-Based Inversion Methods
"""
import os
from src.inversion.Image2StyleGAN_convex_hull import projection as landmark_projection
from src.inversion.Image2StyleGAN import align_image

from src.warping.Warper import get_masks, paste_images
from src.utils.file_utils import generate_file_path, make_dir
from src.morphing.latent_morpher import latent_morpher
from ffhq_dataset.landmarks_detector import LandmarksDetector

landmark_detector = LandmarksDetector("./inversion/weights/shape_predictor_68_face_landmarks.dat")

def landmark_inversion(img1: str, img2: str, network_pkl: str, num_steps: int, morph_coeffs: list, output_dir: str):

    """
    Projects images to StyleGAN latent space using landmark-based inversion and morphs the latents.

    Args:
        img1 (str): Path to the first image.
        img2 (str): Path to the second image.
        network_pkl (str): Path to the StyleGAN model weights.
        num_steps (int): Number of optimization steps for inversion.
        output_dir (str): Directory for output files.
    """

    embeddings_dir = os.path.join(output_dir, "embeddings")
    morphed_dir = os.path.join(output_dir, "morphed")

    # Ensure necessary directories exist
    make_dir(embeddings_dir)
    make_dir(morphed_dir)

    # Generate file paths
    l1 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img1))[0], extension=".npy")
    l2 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img2))[0], extension=".npy")

    # Perform landmark-based inversion
    if not os.path.isfile(l1):
        landmark_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=img1, output_dir=embeddings_dir)

    if not os.path.isfile(l2):
        landmark_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=img2, output_dir=embeddings_dir)


    # Morph Latents
    output_name = f"{os.path.splitext(os.path.basename(img1))[0]}_{os.path.splitext(os.path.basename(img2))[0]}_landmark_no_warping"
    latent_morpher(network_pkl, l1, l2, morph_coeffs, morphed_dir, output_name=output_name)

    return


def landmark_inversion_warping(img1: str, img2: str, network_pkl: str, num_steps: int, morph_coeffs: list, output_dir: str):
    """
    Warps landmarks, performs landmark-based inversion, and morphs the latents.

    Args:
        img1 (str): Path to the first image.
        img2 (str): Path to the second image.
        network_pkl (str): Path to the StyleGAN model weights.
        num_steps (int): Number of optimization steps for inversion.
        output_dir (str): Directory for output files.
    """

    aligned_dir = os.path.join(output_dir, "aligned")
    warped_dir = os.path.join(output_dir, "warped")
    embeddings_dir = os.path.join(output_dir, "embeddings")
    morphed_masks_dir = os.path.join(output_dir, "morphed_masks")

    # Ensure necessary directories exist
    make_dir(aligned_dir)
    make_dir(warped_dir)
    make_dir(embeddings_dir)
    make_dir(morphed_masks_dir)

    # Generate file paths
    aligned1 = generate_file_path(aligned_dir, os.path.splitext(os.path.basename(img1))[0], extension=".png")
    aligned2 = generate_file_path(aligned_dir, os.path.splitext(os.path.basename(img2))[0], extension=".png")
    warped1 = generate_file_path(warped_dir, os.path.splitext(os.path.basename(img1))[0], os.path.splitext(os.path.basename(img2))[0], extension=".png")
    warped2 = generate_file_path(warped_dir, os.path.splitext(os.path.basename(img2))[0], os.path.splitext(os.path.basename(img1))[0], extension=".png")
    l1 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img1))[0], os.path.splitext(os.path.basename(img2))[0], extension=".npy")
    l2 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img2))[0], os.path.splitext(os.path.basename(img1))[0], extension=".npy")
    morphed_mask = generate_file_path(morphed_masks_dir, os.path.splitext(os.path.basename(img1))[0], os.path.splitext(os.path.basename(img2))[0], extension=".png")

    # Align images if not already aligned
    if not os.path.isfile(aligned1):
        align_image(img1, landmark_detector).save(aligned1)
    if not os.path.isfile(aligned2):
        align_image(img2, landmark_detector).save(aligned2)

    # Generate warped masks if not already warped
    if not os.path.isfile(warped1) and not os.path.isfile(warped2):
        get_masks(aligned1, aligned2, warped_dir)

    # Perform landmark-based inversion on warped images
    if not os.path.isfile(l1):
        landmark_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=warped1, output_dir=embeddings_dir)
    if not os.path.isfile(l2):
        landmark_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=warped2, output_dir=embeddings_dir)

    latent_morpher(network_pkl, l1, l2, morph_coeffs, output_dir + "/morphed_masks")


    return