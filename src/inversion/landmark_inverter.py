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
        
    return