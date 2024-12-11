"""
Image2StyleGAN Inversion Method
"""
import os
from src.inversion.Image2StyleGAN import projection as i2s_projection
from src.inversion.Image2StyleGAN import align_image

from src.warping.Warper import get_pasted_masks
from src.morphing.latent_morpher import latent_morpher
from src.utils.file_utils import generate_file_path, make_dir
from ffhq_dataset.landmarks_detector import LandmarksDetector

# Instantiate the landmark detector globally
landmark_detector = LandmarksDetector("./inversion/weights/shape_predictor_68_face_landmarks.dat")

def i2s(img1: str, img2: str, network_pkl: str, num_steps: int, output_dir: str):
    """
    Projects images to StyleGAN latent space using I2S method without any warping and morphs the latents.

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

    # Perform I2S projection for each image if the embedding does not exist
    if not os.path.isfile(l1):
        i2s_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=img1, output_dir=embeddings_dir)

    if not os.path.isfile(l2):
        i2s_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=img2, output_dir=embeddings_dir)

    # Morph Latents
    output_name = f"{os.path.splitext(os.path.basename(img1))[0]}_{os.path.splitext(os.path.basename(img2))[0]}_i2s_no_warping"
    latent_morpher(network_pkl, l1, l2, morphed_dir, output_name=output_name)

    return



def i2s_warping(img1: str, img2: str, network_pkl: str, num_steps: int, output_dir: str):
    """
    Warps landmarks, performs I2S inversion, and morphs the latents.

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
    morphed_dir = os.path.join(output_dir, "morphed")

    # Ensure necessary directories exist
    make_dir(aligned_dir)
    make_dir(warped_dir)
    make_dir(embeddings_dir)
    make_dir(morphed_dir)

    # Generate file paths
    aligned1 = generate_file_path(aligned_dir, os.path.splitext(os.path.basename(img1))[0], extension=".png")
    aligned2 = generate_file_path(aligned_dir, os.path.splitext(os.path.basename(img2))[0], extension=".png")
    warped1 = generate_file_path(warped_dir, os.path.splitext(os.path.basename(img1))[0], os.path.splitext(os.path.basename(img2))[0], extension=".png")
    warped2 = generate_file_path(warped_dir, os.path.splitext(os.path.basename(img2))[0], os.path.splitext(os.path.basename(img1))[0], extension=".png")
    l1 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img1))[0], os.path.splitext(os.path.basename(img2))[0], extension=".npy")
    l2 = generate_file_path(embeddings_dir, os.path.splitext(os.path.basename(img2))[0], os.path.splitext(os.path.basename(img1))[0], extension=".npy")

    # Align images if not already aligned
    if not os.path.isfile(aligned1):
        align_image(img1).save(aligned1)
    if not os.path.isfile(aligned2):
        align_image(img2).save(aligned2)

    # Generate warped images if not already warped
    if not os.path.isfile(warped1) and not os.path.isfile(warped2):
        get_pasted_masks(aligned1, aligned2, warped_dir)

    # Perform I2S inversion on warped images
    if not os.path.isfile(l1):
        i2s_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=warped1, output_dir=embeddings_dir)
    if not os.path.isfile(l2):
        i2s_projection(network_pkl=network_pkl, num_steps=num_steps, input_image=warped2, output_dir=embeddings_dir)

    # Morph Latents
    output_name = f"{os.path.splitext(os.path.basename(img1))[0]}_{os.path.splitext(os.path.basename(img2))[0]}_i2s_with_warping"
    latent_morpher(network_pkl, l1, l2, morphed_dir, output_name=output_name)

    return

