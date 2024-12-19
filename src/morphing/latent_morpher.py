import os
import fnmatch
import numpy as np
import PIL.Image
import torch
import legacy
from torchvision import models

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

image_formats = ["*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def read_dir(dir, formats):
    images = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            for format in formats:
                if fnmatch.fnmatch(name, format):
                    images.append(dir + "/" + name)
    # images.sort()
    return images

def return_max(a1, a2):
    a3 = np.zeros(a1.shape)
    row = 0

    for r1,r2 in zip(a1,a2):
        col = 0
        for v1, v2 in zip(r1, r2):
            if v1 >= v2:
                a3[row][col] = v1
            else:
                a3[row][col] = v2
            col += 1
        row += 1
    return a3

def return_min(a1, a2):
    a3 = np.zeros(a1.shape)
    row = 0

    for r1,r2 in zip(a1,a2):
        col = 0
        for v1, v2 in zip(r1, r2):
            if v1 <= v2:
                a3[row][col] = v1
            else:
                a3[row][col] = v2
            col += 1
        row += 1
    return a3 * 0.5

def latent_morpher(network_pkl, l1, l2, morph_coeffs, output_dir, output_name = ""):
    """
    Performs latent morphing for given coefficients and saves the results.

    Args:
        network_pkl (str): Path to the StyleGAN model weights.
        l1 (str): Path to the latent vector of the first image.
        l2 (str): Path to the latent vector of the second image.
        morph_coeffs (list): List of morphing coefficients.
        output_dir (str): Directory to save the morphed images.
    """
    with src.dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)


    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name } # dictionary 17 : tensor(4,4), ...  ,tensor(1024,1024)

    random_noise = []
    for buf in noise_bufs.values():
        random_noise.append((torch.randn_like(buf).to(device) * 0.7))


    latent1 = np.load(l1)  # (1, 18, 512)
    latent2 = np.load(l2)  # (1, 18, 512)

    for coeff in morph_coeffs:
        starting_latent = (coeff * latent1 + (1 - coeff) * latent2)

        s_l = torch.tensor(starting_latent, dtype=torch.float32, device=device, requires_grad=False).to(device)
        synth_image = G.synthesis(s_l, noise_mode='const')
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        # Save image with coefficient in the name
        coeff_str = f"{coeff:.2f}"
        output_name_full = f"{output_name}_coeff_{coeff_str}.png"

        if len(morph_coeffs) == 1:
            output_name_full = l1.split("/")[-1].split(".")[0] + '.png'

        PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(output_dir, output_name_full))


class VGG16_perceptual(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(VGG16_perceptual, self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(2)])
        self.slice2 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(2, 4)])
        self.slice3 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 14)])
        self.slice4 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(14, 21)])

        # Freeze parameters if gradient computation is not required
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Extracts feature maps from different layers of VGG16.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].

        Returns:
            tuple[torch.Tensor, ...]: Feature maps from four layers of VGG16.
        """
        h = x
        features = []
        for slice in self.slices:
            h = slice(h)
            features.append(h)
        return tuple(features)

