# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation,
# and any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modifications:
# This script has been modified.
# Key changes include:
# - Integrated utility functions from `src.utils` for improved modularity.
# - Replaced hardcoded VGG16 loading logic with `VGG16_perceptual` from `src.morphing`.
# - Adjust noise regularization.
# - Added `align_image` for target image preprocessing.
# - Improved documentation for better maintainability.


import copy
import os
import sys
from time import perf_counter
import click
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import dnnlib
import legacy
from src.utils import align_image, read_dir
from src.morphing import VGG16_perceptual
from ffhq_dataset.landmarks_detector import LandmarksDetector


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Instantiate the landmark detector globally
landmark_detector = LandmarksDetector("./inversion/weights/shape_predictor_68_face_landmarks.dat")

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    euclidean_dist_weight      = 0.0,
    pixel_weight               = 0.01,
    mtcnn_weight               = 1,
    regularize_mag_weight      = 0.1,
    dist_weight                = 1,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    vgg_weights                = 'weights/vgg16.pt',
    output                     = "",
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # light_loss = light_cnn_loss()
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    vgg16 = VGG16_perceptual().to(device)

    target_images = target.unsqueeze(0).to(device).to(torch.float32)

    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')

    # Feature Maps from the 4 different layers of VGG16
    tf_0, tf_1, tf_2, tf_3 = vgg16(target_images)

    w_opt = torch.tensor(w_avg.repeat(18, axis=1), dtype=torch.float32, device=device, requires_grad=True).to(device) # pylint: disable=not-callable
    w_out = torch.zeros([0] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    # Adam Optimizer
    optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    pixelwise_loss = torch.nn.L1Loss()

    mse = torch.nn.MSELoss()

    original_image = target.unsqueeze(0).to(device).to(torch.float32)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)