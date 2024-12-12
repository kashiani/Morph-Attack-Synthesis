import imageio
import os
import argparse
from pathlib import Path

def frames_to_gif(input_directory, output_directory, frame_duration):
    # Convert to Path object for easier manipulation
    input_directory = Path(input_directory)
    if not input_directory.is_dir():
        raise ValueError(f"The input directory {input_directory} does not exist or is not a directory.")

    # Ensure the output directory exists; if not, create it
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Extract the specific part of the path for the output filename
    # Assuming the desired part is always one level up from the input directory
    project_name = input_directory.parent.name
    output_filename = f"{project_name}_output.gif"
    output_path = output_directory / output_filename

    # List all image files in the input directory
    frame_files = sorted(
        [file for file in input_directory.glob('*') if file.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    # Read the frames from files
    frames = [imageio.imread(file) for file in frame_files]

    # Write the frames to a GIF file
    imageio.mimsave(output_path, frames, 'GIF', duration=frame_duration)


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
