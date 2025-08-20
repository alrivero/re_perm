import cv2
import numpy as np
import argparse
import os
import csv
from pathlib import Path

def crop_with_padding(image, x1, y1, crop_size):
    """
    Crops a region from an image based on top-left coordinates and size.

    If the specified crop region extends beyond the image boundaries (as it
    might if the new image is smaller than the original), the output is
    padded with black pixels (0) to maintain the target crop_size.

    Args:
        image (np.ndarray): The source image to crop from.
        x1 (int): The target top-left x-coordinate of the crop.
        y1 (int): The target top-left y-coordinate of the crop.
        crop_size (int): The side length of the square crop.

    Returns:
        np.ndarray: The cropped (and possibly padded) image.
    """
    img_h, img_w = image.shape[:2]

    # Create a blank canvas of the desired crop size (in color)
    if len(image.shape) == 3:
        canvas = np.zeros((crop_size, crop_size, image.shape[2]), dtype=image.dtype)
    else: # Grayscale image
        canvas = np.zeros((crop_size, crop_size), dtype=image.dtype)

    # Calculate the source region (from the input image)
    # This is the part of the crop box that is actually inside the image
    src_x_start = max(0, x1)
    src_y_start = max(0, y1)
    src_x_end = min(img_w, x1 + crop_size)
    src_y_end = min(img_h, y1 + crop_size)

    # Calculate the destination region (on the new canvas)
    # This is where the valid part of the source will be pasted
    dst_x_start = max(0, -x1)
    dst_y_start = max(0, -y1)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # Copy the valid region from the source image to the destination canvas
    if (src_x_end > src_x_start) and (src_y_end > src_y_start):
        canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]

    return canvas


def main():
    """Main function to parse arguments and apply crops."""
    parser = argparse.ArgumentParser(
        description="Applies pre-computed cropping parameters from a log file to a new set of images.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input_dir', type=str, required=True,
        help='Path to the new directory of images to be cropped.'
    )
    parser.add_argument(
        '-l', '--log_file', type=str, required=True,
        help='Path to the crop_log.csv file containing the crop parameters.'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True,
        help='Path to the directory where new cropped images will be saved.'
    )
    parser.add_argument(
        '-r', '--resize', type=int,
        help='Optional: Resize the final crop to R x R pixels.'
    )
    args = parser.parse_args()

    # --- Setup ---
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    log_path = Path(args.log_file)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Read Log File ---
    if not log_path.exists():
        print(f"❌ Error: Log file not found at '{log_path}'")
        return
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        # Create a dictionary mapping original_filename to its log data for easy lookup
        log_map = {row['original_filename']: row for row in reader}

    if not log_map:
        print(f"❌ Error: Log file '{log_path}' is empty or could not be read.")
        return

    # --- Get Input Files ---
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    input_files = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]

    if not input_files:
        print(f"❌ No images found in the input directory '{input_path}'.")
        return

    print(f"Found {len(log_map)} log entries and {len(input_files)} images to process.")
    
    # --- Processing Loop ---
    files_processed = 0
    for image_path in input_files:
        # Check if the current image has an entry in the log file
        if image_path.name not in log_map:
            print(f"⚠️ Warning: No log entry found for '{image_path.name}'. Skipping.")
            continue

        try:
            log_entry = log_map[image_path.name]

            # --- 1. Parse Log Data ---
            x1 = int(log_entry['top_left_x'])
            y1 = int(log_entry['top_left_y'])
            crop_size = int(log_entry['crop_size_before_resize'])

            # --- 2. Load the Image ---
            image_to_crop = cv2.imread(str(image_path))
            if image_to_crop is None:
                print(f"⚠️ Warning: Could not read image '{image_path.name}'. Skipping.")
                continue

            # --- 3. Crop the Image using the robust padding function ---
            cropped_image = crop_with_padding(image_to_crop, x1, y1, crop_size)

            # --- 4. Resize if Requested ---
            if args.resize:
                # For natural images, INTER_AREA is good for shrinking, INTER_CUBIC for enlarging.
                interp = cv2.INTER_AREA if args.resize < crop_size else cv2.INTER_CUBIC
                final_image = cv2.resize(cropped_image, (args.resize, args.resize), interpolation=interp)
            else:
                final_image = cropped_image

            # --- 5. Save Result ---
            output_filename = f"{image_path.stem}.png"
            output_filepath = output_path / output_filename
            cv2.imwrite(str(output_filepath), final_image)
            files_processed += 1

        except Exception as e:
            print(f"❌ Error processing '{image_path.name}': {e}")
            
    print(f"\n✅ Processing complete. {files_processed} images were cropped and saved in '{output_path}'.")


if __name__ == '__main__':
    main()