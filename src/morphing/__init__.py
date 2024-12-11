"""
Morphing module initialization.

This module provides:
- Latent morphing functionality.
- Perceptual loss utilities using VGG16.
- Helper functions for max and min operations on arrays.
"""

from .latent_morpher import (
    latent_morpher,
    VGG16_perceptual,
    return_max,
    return_min,
)

__all__ = [
    "latent_morpher",
    "VGG16_perceptual",
    "return_max",
    "return_min",
]
