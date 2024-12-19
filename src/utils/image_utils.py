import os
import torch
import numpy as np
import PIL
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


def align_image(image_path: str, landmark_detector: LandmarksDetector) -> PIL.Image.Image:
    """
    Aligns a face image using detected landmarks.

    Args:
        image_path (str): Path to the input image.
        landmark_detector (LandmarksDetector): A landmark detection model instance.

    Returns:
        PIL.Image.Image: Aligned image.
    """
    for landmarks in landmark_detector.get_landmarks(image_path):
        aligned_img = image_align(image_path, "", landmarks)
        print(f"Aligned Image: {image_path}")
        return aligned_img
    raise ValueError(f"No landmarks detected in the image: {image_path}")
