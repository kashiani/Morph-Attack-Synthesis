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
