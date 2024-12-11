"""
Main entry point for the StyleGAN Morphing project.

Handles the command-line interface and orchestrates various modules
for inversion, warping, and latent morphing.

Features:
- Supports Image-to-StyleGAN (I2S) and Landmark-based inversion.
- Optional warping of landmarks before inversion.
"""

import os
import argparse
from tqdm import tqdm
import warnings
from src.inversion.image_inverter import i2s, i2s_warping
from src.inversion.landmark_inverter import landmark_inversion, landmark_inversion_warping
from src.utils.file_utils import make_dir



def process_pair(pair: str, method: str, inversion: str, warping: bool, network_pkl: str, num_steps: int, morph_coeffs: list, output_dir: str):
    """
    Process a pair of images using the specified synthesis method and inversion/warping methods.

    Args:
        pair (str): The pair of image paths separated by '|'.
        method (str): Morph synthesis method ('StyleGAN', 'OpenCV', 'FaceMorpher').
        inversion (str): Inversion type ('I2S' or 'Landmark').
        warping (bool): Whether to apply warping.
        network_pkl (str): Path to the StyleGAN model weights.
        num_steps (int): Number of optimization steps for inversion.
        morph_coeffs (list): List of morphing coefficients (e.g., [0.2, 0.5, 0.8]).
        output_dir (str): Directory for output files.
    """
    img1, img2 = pair.strip().split(" | ")
    name1 = os.path.splitext(os.path.basename(img1))[0]
    name2 = os.path.splitext(os.path.basename(img2))[0]

    # Create subdirectory for this pair
    warping_status = "Warping" if warping else "NoWarping"
    pair_output_dir = os.path.join(output_dir, f"{method}_{inversion}_{warping_status}_{name1}_{name2}")
    make_dir(pair_output_dir)

    if method == "StyleGAN":
        # Ensure necessary subdirectories exist for StyleGAN
        make_dir(os.path.join(pair_output_dir, "embeddings"))
        make_dir(os.path.join(pair_output_dir, "morphed"))

        if inversion == "I2S" and not warping:
            i2s(img1, img2, network_pkl, num_steps, morph_coeffs, pair_output_dir)
        elif inversion == "I2S" and warping:
            make_dir(os.path.join(pair_output_dir, "warped"))
            make_dir(os.path.join(pair_output_dir, "aligned"))
            make_dir(os.path.join(pair_output_dir, "morphed_masks"))
            i2s_warping(img1, img2, network_pkl, num_steps, morph_coeffs, pair_output_dir)
        elif inversion == "Landmark" and not warping:
            landmark_inversion(img1, img2, network_pkl, num_steps, morph_coeffs, pair_output_dir)
        elif inversion == "Landmark" and warping:
            if len(morph_coeffs) != 1:
                raise ValueError(
                    "For 'Landmark' inversion with warping, morph_coeffs must contain exactly one coefficient.")

            make_dir(os.path.join(pair_output_dir, "warped"))
            make_dir(os.path.join(pair_output_dir, "aligned"))
            make_dir(os.path.join(pair_output_dir, "morphed_masks"))
            landmark_inversion_warping(img1, img2, network_pkl, num_steps, morph_coeffs, pair_output_dir)
    elif method == "OpenCV":
        print(f"Processing {name1} and {name2} using OpenCV morphing method...")
        # TODO: Implement OpenCV-based morph synthesis here
        pass
    elif method == "FaceMorpher":
        print(f"Processing {name1} and {name2} using FaceMorpher method...")
        # TODO: Implement FaceMorpher-based morph synthesis here
        pass
    else:
        raise ValueError(f"Unknown morphing method: {method}")




def main():
    """
    Command-line interface for the StyleGAN Morphing project.
    """
    parser = argparse.ArgumentParser(description="StyleGAN Morphing Tool")

    # Command-line arguments
    parser.add_argument('--list', type=str, default="./examples/morph_pairs.txt", help="Path to the list of morph pairs.")
    parser.add_argument('--inversion', type=str, choices=["I2S", "Landmark"], default="I2S", help="Inversion method to use.")
    parser.add_argument('--warping', action='store_true', help="Apply warping to landmarks before inversion.")

    parser.add_argument('--network', type=str, default="./inversion/weights/ffhq.pkl", help="Path to the StyleGAN model weights.")
    parser.add_argument('--num_steps', type=int, default=1000, help="Number of optimization steps for inversion.")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory for all generated data.")

    args = parser.parse_args()

    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Failed to build CUDA kernels for upfirdn2d")
    warnings.filterwarnings("ignore", category=UserWarning)


    # Read morph pairs
    with open(args.list, 'r') as f:
        pairs = f.readlines()


    # Process each pair with a progress bar
    for pair in tqdm(pairs, desc="Processing morph pairs", unit="pair"):
        process_pair(pair, args.inversion, args.warping, args.network, args.num_steps, args.output_dir)


if __name__ == "__main__":
    main()
