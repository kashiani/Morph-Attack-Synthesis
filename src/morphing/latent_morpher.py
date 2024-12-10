import os
import fnmatch
import numpy as np
import PIL.Image
import torch
import dnnlib
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