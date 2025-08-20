import cv2
import os
import argparse
import shutil
import time
from pathlib import Path

def main():
    """Main function to parse arguments and resize images."""
    parser = argparse.ArgumentParser(
        description="Resizes all images in a directory, overwriting the original files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-d', '--dir', type=str, required=True,
        help='Path to the directory containing images to resize.'
    )
    parser.add_argument(
        '-w', '--width', type=int, required=True,
        help='The target width for the resized images.'
    )
    parser.add_argument(
        '-ht', '--height', type=int, required=True,
        help='The target height for the resized images.'
    )
    parser.add_argument(
        '--backup', action='store_true',
        help='Create a backup of the original images before resizing.'
    )
    args = parser.parse_args()

    # --- Setup ---
    target_dir = Path(args.dir)
    target_resolution = (args.width, args.height)
    backup_dir = None

    if not target_dir.is_dir():
        print(f"❌ Error: Directory not found at '{target_dir}'")
        return

    # Supported image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    files_to_process = [p for p in target_dir.iterdir() if p.suffix.lower() in image_extensions]

    if not files_to_process:
        print(f"No images found in '{target_dir}'.")
        return

    print(f"Found {len(files_to_process)} images to resize to {args.width}x{args.height} pixels.")
    
    # --- Create Backup Directory if Flagged ---
    if args.backup:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = target_dir / f"backups_{timestamp}"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Backing up original files to '{backup_dir}'")
        except OSError as e:
            print(f"❌ Error creating backup directory: {e}")
            return # Exit if backup fails

    # --- Processing Loop ---
    for image_path in files_to_process:
        try:
            # --- Backup File ---
            if args.backup and backup_dir:
                shutil.copy(str(image_path), str(backup_dir))

            # --- Read and Resize Image ---
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️ Warning: Could not read {image_path.name}. Skipping.")
                continue

            # Choose interpolation based on whether we are shrinking or enlarging
            current_height, current_width = img.shape[:2]
            if target_resolution[0] < current_width or target_resolution[1] < current_height:
                interpolation = cv2.INTER_AREA # Best for shrinking
            else:
                interpolation = cv2.INTER_CUBIC # Best for enlarging

            resized_img = cv2.resize(img, target_resolution, interpolation=interpolation)

            # --- Overwrite Original File ---
            cv2.imwrite(str(image_path), resized_img)
            print(f"Resized {image_path.name}")

        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")

    print(f"\n✅ Done. All images have been resized.")

if __name__ == '__main__':
    main()