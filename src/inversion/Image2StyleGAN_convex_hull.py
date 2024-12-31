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

    # Prepare the generator
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # Compute W midpoint and standard deviation
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)      # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Initialize noise buffers
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector
    if vgg_weights is not None:
        vgg_weights = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(vgg_weights) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Extract features for the target image
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # Initialize latent variables
    w_opt = torch.tensor(w_avg.repeat(18, 1), dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([0] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    # Setup optimizer
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    pixelwise_loss = torch.nn.L1Loss()

    # Initialize noise buffers
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Optimization loop
    for step in range(num_steps):
        # Learning rate schedule
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp *= min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synthesize images from latent variables
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_images = G.synthesis(ws, noise_mode='const')
        synth_images = (synth_images + 1) * (255 / 2)

        # Compute pixel-wise loss
        pixel_loss = pixelwise_loss(target_images, synth_images.clamp(0, 255))

        # Downsample synthesized images if necessary
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Compute VGG feature distance
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum() * dist_weight

        # Compute noise regularization loss
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        # Compute total loss
        loss = dist + reg_loss * regularize_noise_weight + pixel_loss * pixel_weight
        if regularize_mag_weight > 0:
            latent_mag_reg = (torch.mean(torch.square(ws)) ** 0.5)
            loss += latent_mag_reg * regularize_mag_weight

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Print progress
        logprint(f'Step {step + 1}/{num_steps}: Loss {loss.item():.4f}')

        # Normalize noise buffers
        with torch.no_grad():
            for buf in noise_bufs.values():
                if step > 400:
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt() * 0
                else:
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    return
