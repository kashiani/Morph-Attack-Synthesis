import os
import fnmatch
from typing import List
def read_dir(directory: str, formats: List[str]) -> List[str]:
    """
    Reads a directory and returns a list of image file paths matching specified formats.

    Args:
        directory (str): The directory to scan.
        formats (list[str]): List of acceptable file formats (e.g., ['*.jpg', '*.png']).

    Returns:
        list[str]: List of paths to matching image files.
    """
    images = []
    for root, _, files in os.walk(directory):
        for name in files:
            if any(fnmatch.fnmatch(name, fmt) for fmt in formats):
                images.append(os.path.join(root, name))
    return images


def ensure_dir_exists(path):
    """Ensure that a directory exists; create if not.

    Args:
        path (str): Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def generate_file_path(base_dir: str, *parts: str, extension: str) -> str:
    """
    Generates a file path by combining a base directory, additional parts, and a file extension.

    Args:
        base_dir (str): The base directory for the file.
        *parts (str): Additional parts to construct the file name.
        extension (str): The file extension (e.g., '.png', '.npy').

    Returns:
        str: The constructed file path.
    """
    file_name = "_".join(parts) + extension
    return os.path.join(base_dir, file_name)


def make_dir(path: str):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
