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

