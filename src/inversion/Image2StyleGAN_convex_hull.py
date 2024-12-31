# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Script Name: Image2StyleGAN_facial_convex_hull.py

Description:
This script performs targeted facial manipulation using the convex hull of a face. It uses a pretrained
StyleGAN to project a given image into its latent space, reconstructs the entire image, and subsequently
isolates the facial convex hull. The facial region is then seamlessly pasted back into the original image,
ensuring that only the convex hull area is affected while preserving the natural background and hair.

Key Features:
1. **Latent Space Projection and Whole Image Reconstruction**:
    - The initial latent code is computed for the entire image.
    - The StyleGAN generates a reconstructed version of the complete input image based on the latent code.

2. **Convex Hull Isolation and Reintegration**:
    - After reconstruction, the facial convex hull is extracted using landmarks.
    - The convex hull region is pasted back into the original input image, replacing only the corresponding area.

3. **Landmark-Based Convex Hull Extraction**:
    - Uses a facial landmark detector to identify key points for convex hull extraction.
    - Guarantees that inversion and reconstruction only modify the face, leaving hair and background intact.

4. **Optimization and Loss Functions**:
    - Combines perceptual loss, pixel-wise loss, and noise regularization for accurate latent code optimization.
    - Utilizes adam optimization to ensure efficient convergence.

"""


import copy
import os
from time import perf_counter
import fnmatch
import click
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import dnnlib
import sys
import legacy
from src.utils import align_image, read_dir
from ffhq_dataset.landmarks_detector import LandmarksDetector
from facenet_pytorch import MTCNN as facenet_mtcnn, InceptionResnetV1




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Instantiate the landmark detector globally
landmark_detector = LandmarksDetector("./inversion/weights/shape_predictor_68_face_landmarks.dat")


def torch_l2(a, b):
    return (a - b).pow(2).sum().sqrt()

def relativeLandmarkLoss(l1, l2):
    target_left_eye = ((l1[0][0][0] - l1[0][2][0]).pow(2) + (l1[0][0][1] - l1[0][2][1]).pow(2)).sqrt()
    target_right_eye = ((l1[0][1][0] - l1[0][2][0]).pow(2) + (l1[0][1][1] - l1[0][2][1]).pow(2)).sqrt()
    target_left_mouth = ((l1[0][3][0] - l1[0][2][0]).pow(2) + (l1[0][3][1] - l1[0][2][1]).pow(2)).sqrt()
    target_right_mouth = ((l1[0][4][0] - l1[0][2][0]).pow(2) + (l1[0][4][1] - l1[0][2][1]).pow(2)).sqrt()

    synth_left_eye = ((l2[0][0][0] - l2[0][2][0]).pow(2) + (l2[0][0][1] - l2[0][2][1]).pow(2)).sqrt()
    synth_right_eye = ((l2[0][1][0] - l2[0][2][0]).pow(2) + (l2[0][1][1] - l2[0][2][1]).pow(2)).sqrt()
    synth_left_mouth = ((l2[0][3][0] - l2[0][2][0]).pow(2) + (l2[0][3][1] - l2[0][2][1]).pow(2)).sqrt()
    synth_right_mouth = ((l2[0][4][0] - l2[0][2][0]).pow(2) + (l2[0][4][1] - l2[0][2][1]).pow(2)).sqrt()

    left_eye = torch_l2(target_left_eye, synth_left_eye)
    right_eye = torch_l2(target_right_eye, synth_right_eye)
    left_mouth = torch_l2(target_left_mouth, synth_left_mouth)
    right_mouth = torch_l2(target_right_mouth, synth_right_mouth)
    return left_eye + right_eye + left_mouth + right_mouth




def project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.3,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    euclidean_dist_weight=0.0,
    pixel_weight=0.01,
    mtcnn_weight=1,
    regularize_mag_weight=0.1,
    dist_weight=1,
    regularize_noise_weight=1e5,
    verbose=False,
    vgg_weights='weights/vgg16.pt',
    output_name="",
    device: torch.device
):
    """
    Optimize latent variables in a generative model to approximate a target image.

    Args:
        G: Pre-trained generative model.
        target (torch.Tensor): Target image tensor with shape [C, H, W] in range [0, 255].
        num_steps (int): Number of optimization steps.
        w_avg_samples (int): Number of samples for computing W midpoint and stddev.
        initial_learning_rate (float): Initial learning rate for optimization.
        initial_noise_factor (float): Initial scale of noise added to latent variables.
        lr_rampdown_length (float): Ramp-down length for learning rate schedule.
        lr_rampup_length (float): Ramp-up length for learning rate schedule.
        noise_ramp_length (float): Ramp length for noise scaling.
        euclidean_dist_weight (float): Weight for Euclidean distance loss (not used in the code).
        pixel_weight (float): Weight for pixel-wise loss.
        mtcnn_weight (float): Weight for MTCNN-based loss (not used in the code).
        regularize_mag_weight (float): Weight for latent magnitude regularization.
        dist_weight (float): Weight for VGG-based feature distance loss.
        regularize_noise_weight (float): Weight for noise regularization.
        verbose (bool): If True, prints progress messages.
        vgg_weights (str): Path to VGG16 weights.
        output_name (str): Name for output files.
        device (torch.device): Torch device to use (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """

    # Validate target image shape
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), \
        "Target image dimensions do not match the generator output resolution."

    def logprint(*args):
        if verbose:
            print(*args)



    return