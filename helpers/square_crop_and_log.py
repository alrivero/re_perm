import cv2
import numpy as np
import argparse
import os
import csv
from pathlib import Path

def main():
    # Argument parser setup (remains the same)
    parser = argparse.ArgumentParser(
        description=(
            "Crop binary masks from a directory. The crop is a square centered on the mask's\n"
            "center of gravity and is maximized to extend to the image borders."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (all arguments are identical to the previous version)
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to the directory containing binary mask images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Path to the directory where cropped images will be saved.')
    parser.add_argument('-r', '--resize', type=int, help='Optional: Resize the final crop to R x R pixels.')
    parser.add_argument('--ignore_x', action='store_true', help="Ignore the mask's center of gravity X-axis, using the image's horizontal center instead.")
    parser.add_argument('--ignore_y', action='store_true', help="Ignore the mask's center of gravity Y-axis, using the image's vertical center instead.")
    parser.add_argument('--log_file', type=str, default='crop_log.csv', help='Name of the CSV file to record crop positions. Default: crop_log.csv')
    args = parser.parse_args()

    # --- Setup ---
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    files_to_process = [p for p in input_path.iterdir() if p.suffix.lower() in image_extensions]
    if not files_to_process:
        print(f"❌ No images found in '{input_path}'.")
        return
    print(f"Found {len(files_to_process)} images to process...")

    log_filepath = output_path / args.log_file
    with open(log_filepath, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        # --- MODIFICATION START ---
        # Add original_width and original_height to the header
        header = [
            'original_filename', 'output_filename', 'original_width', 'original_height',
            'top_left_x', 'top_left_y', 'crop_size_before_resize'
        ]
        # --- MODIFICATION END ---
        csv_writer.writerow(header)

        # --- Processing Loop ---
        for image_file in files_to_process:
            try:
                mask = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"⚠️ Warning: Could not read {image_file.name}. Skipping.")
                    continue
                
                img_h, img_w = mask.shape

                # Center finding logic (remains the same)
                moments = cv2.moments(mask)
                if moments['m00'] == 0:
                    print(f"⚠️ Warning: Mask {image_file.name} is empty (all zeros). Skipping.")
                    continue
                cg_x = int(moments['m10'] / moments['m00'])
                cg_y = int(moments['m01'] / moments['m00'])
                center_x = img_w // 2 if args.ignore_x else cg_x
                center_y = img_h // 2 if args.ignore_y else cg_y

                # Max crop calculation (remains the same)
                half_size = min(center_x, img_w - center_x, center_y, img_h - center_y)
                crop_size = half_size * 2
                if crop_size == 0:
                    print(f"⚠️ Warning: Cannot create a crop for {image_file.name}. Skipping.")
                    continue
                
                # Cropping and resizing (remains the same)
                top_left_x = center_x - half_size
                top_left_y = center_y - half_size
                cropped_mask = mask[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]
                final_mask = cv2.resize(cropped_mask, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST) if args.resize else cropped_mask
                
                # Saving (remains the same)
                output_filename = f"{image_file.stem}{image_file.suffix}"
                cv2.imwrite(str(output_path / output_filename), final_mask)
                
                # --- MODIFICATION START ---
                # Add the new dimension data to the log row
                csv_writer.writerow([
                    image_file.name,
                    output_filename,
                    img_w,          # Add original width
                    img_h,          # Add original height
                    top_left_x,
                    top_left_y,
                    crop_size
                ])
                # --- MODIFICATION END ---

            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")

    print(f"\n✅ Processing complete. Cropped images are in '{output_path}'.")
    print(f"📄 Crop positions logged in '{log_filepath}'.")

if __name__ == '__main__':
    main()