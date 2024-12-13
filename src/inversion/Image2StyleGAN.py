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

    for step in range(num_steps):

        # Synth images from opt_w.
        synth_images = G.synthesis(w_opt, noise_mode='const')
        synth_images = (synth_images + 1) * (255/2)

        # Pixel-wise loss
        pix = mse(original_image, synth_images)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        sf_0, sf_1, sf_2, sf_3 = vgg16(synth_images)
        # Perceptual Loss
        dist = 0
        dist += mse(tf_0, sf_0)
        dist += mse(tf_1, sf_1)
        dist += mse(tf_2, sf_2)
        dist += mse(tf_3, sf_3)

        # Total Loss
        loss = dist + pix

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(f'step {step+1:>4d}/{num_steps}: VGG- {dist:<4.2f} Pixel- {pix:<4.2f} Total- {float(loss):<5.2f}', end="\r")
        sys.stdout.write("\033[K")

        # Save projected W for each optimization step.
        w_out = w_opt.detach()

    # Save output image and latent code
    synth_image = G.synthesis(w_out, noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(output + '.png')

    np.save(output + '.npy', w_out.cpu().numpy())

    return

# Project Funtion
def projection(network_pkl, num_steps, input_image, output_dir, seed=303):
    np.random.seed(seed)
    torch.manual_seed(seed)

    image = input_image
    output_dir = output_dir + "/"
    with dnnlib.util.open_url(network_pkl) as fp:
        # Load networks.
        print('Loading networks from "%s"...' % network_pkl)
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

        print('Embedding', image)
        name = image.split("/")[-1].split(".")[0]
        target_fname = image

        # Load target image.
        target_pil = align_image(image, landmark_detector)# .convert('RGB')

        # Save Aligned Image
        target_pil.save(output_dir + name + '_aligned.png')

        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
        start_time = perf_counter()
        project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            num_steps=num_steps,
            initial_learning_rate=0.01,
            device=device,
            verbose=True,
            vgg_weights='weights/vgg16.pt',
            output = output_dir + name
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', default='weights/ffhq.pkl', help='Network pickle filename')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--input_image',            help='Image to Invert', type=str, default="input/04203d523.jpg")
@click.option('--output_dir',             help='Output Folder for Latent and Reconstructed Image', type=str, default="output")

def run_projection(network_pkl: str, seed: int, num_steps: int, input_image: str, output_dir: str):

    np.random.seed(seed)
    torch.manual_seed(seed)

    image = input_image
    output_dir = output_dir + "/"
    with dnnlib.util.open_url(network_pkl) as fp:
        # Load networks.
        print('Loading networks from "%s"...' % network_pkl)
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

        print('Embedding', image)
        name = image.split("/")[-1].split(".")[0]
        target_fname = image

        # Load target image.
        target_pil = align_image(image)# .convert('RGB')

        # Save Aligned Image
        target_pil.save(output_dir + name + '_aligned.png')


        target_uint8 = np.array(target_pil, dtype=np.uint8)

        # Optimize projection.
        start_time = perf_counter()
        project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            num_steps=num_steps,
            initial_learning_rate=0.01,
            device=device,
            verbose=True,
            vgg_weights='weights/vgg16.pt',
            output = output_dir + name
        )
        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

if __name__ == "__main__":
    run_projection()


