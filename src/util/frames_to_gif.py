import imageio
import os
import argparse
from pathlib import Path

def frames_to_gif(input_directory, output_directory, frame_duration):
    # Convert to Path object for easier manipulation
    input_directory = Path(input_directory)
    if not input_directory.is_dir():
        raise ValueError(f"The input directory {input_directory} does not exist or is not a directory.")



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert image frames to GIF.")
    parser.add_argument('--input_directory', type=str, default = '', help="Directory containing image frames.")
    parser.add_argument('--output_path', type=str, default = '', help="Output path for the GIF file.")
    parser.add_argument('--frame_duration', type=float, default=0.1,
                        help="Duration each frame is displayed in the GIF, in seconds.")

    # Parse arguments
    args = parser.parse_args()

    # Run the frames to GIF conversion
    frames_to_gif(args.input_directory, args.output_path, args.frame_duration)

if __name__ == '__main__':
    main()
