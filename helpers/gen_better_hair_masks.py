import cv2
import numpy as np
from skimage.measure import label
import os
import argparse

def clean_mask_with_hair(hair_mask_path, full_body_mask_path, dilation_kernel_size=5, beard_mask_path=None, beard_dilation_kernel_size=0, body_dilation_iterations=1):
    """
    Cleans a full-body mask by removing hair and other artifacts.

    Args:
        hair_mask_path (str): The file path for the hair mask image.
        full_body_mask_path (str): The file path for the full-body mask image.
        dilation_kernel_size (int, optional): The size of the kernel for body dilation. Defaults to 5.
        beard_mask_path (str, optional): The file path for the optional beard mask. Defaults to None.
        beard_dilation_kernel_size (int, optional): Kernel size to dilate the beard mask. Defaults to 0 (no dilation).
        body_dilation_iterations (int, optional): Number of iterations for body dilation. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - cleaned_mask (numpy.ndarray): The intermediate cleaned mask.
            - final_mask (numpy.ndarray): The final mask after all operations.
    """

    # --- Step 1: Load and Prepare Masks ---
    # Load the images in grayscale
    hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
    full_body_mask = cv2.imread(full_body_mask_path, cv2.IMREAD_GRAYSCALE)

    # Handle cases where images might not be found or are invalid
    if hair_mask is None:
        print(f"Warning: Could not read hair mask at {hair_mask_path}. Skipping.")
        return None, None
    if full_body_mask is None:
        print(f"Warning: Could not read full body mask at {full_body_mask_path}. Skipping.")
        return None, None

    # Ensure the masks are binary (0 or 255)
    _, hair_mask_bin = cv2.threshold(hair_mask, 127, 255, cv2.THRESH_BINARY)
    _, full_body_mask_bin = cv2.threshold(full_body_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Optional Step: Process and Combine Beard Mask ---
    if beard_mask_path and os.path.exists(beard_mask_path):
        beard_mask = cv2.imread(beard_mask_path, cv2.IMREAD_GRAYSCALE)
        if beard_mask is not None:
            _, beard_mask_bin = cv2.threshold(beard_mask, 127, 255, cv2.THRESH_BINARY)

            # Dilate the beard mask if a kernel size is provided
            if beard_dilation_kernel_size > 0:
                beard_kernel = np.ones((beard_dilation_kernel_size, beard_dilation_kernel_size), np.uint8)
                beard_mask_bin = cv2.dilate(beard_mask_bin, beard_kernel, iterations=1)

            # XOR the hair mask with the (potentially dilated) beard mask.
            hair_mask_bin = cv2.bitwise_xor(hair_mask_bin, beard_mask_bin)
        else:
            print(f"Warning: Could not read beard mask at {beard_mask_path}. Proceeding without it.")


    # --- Step 2: Initial XOR Operation ---
    # XOR the full-body mask with the (potentially combined) hair mask
    xored_mask = cv2.bitwise_xor(full_body_mask_bin, hair_mask_bin)

    # --- Step 3: Find and Isolate the Largest Connected Component ---
    labels = label(xored_mask)
    if labels.max() == 0:
        print(f"Warning: No connected components found for {os.path.basename(hair_mask_path)}. Returning an empty mask.")
        return np.zeros_like(xored_mask), np.zeros_like(xored_mask)

    bincount = np.bincount(labels.flat)
    if len(bincount) <= 1:
        return np.zeros_like(xored_mask), np.zeros_like(xored_mask)
        
    largest_cc_label = np.argmax(bincount[1:]) + 1
    largest_cc_mask = (labels == largest_cc_label).astype(np.uint8) * 255

    # --- Step 4: Dilate the Largest Component ---
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(largest_cc_mask, kernel, iterations=body_dilation_iterations)

    # --- Step 5: Create the Cleaned Mask ---
    cleaned_mask = cv2.bitwise_and(xored_mask, xored_mask, mask=dilated_mask)

    # --- Step 6: Dilate the Cleaned Mask ---
    if dilation_kernel_size > 0 and body_dilation_iterations > 0:
        cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=body_dilation_iterations)

    # --- Step 7: Intermediate XOR Operation ---
    intermediate_final_mask = cv2.bitwise_xor(cleaned_mask, full_body_mask_bin)

    # --- Step 8: Post-processing on the Final Mask ---
    # Find the largest component in the intermediate final mask
    final_labels = label(intermediate_final_mask)
    if final_labels.max() == 0:
        # If the intermediate mask is empty, the final mask should also be empty
        return cleaned_mask, np.zeros_like(intermediate_final_mask)

    final_bincount = np.bincount(final_labels.flat)
    if len(final_bincount) <= 1:
        return cleaned_mask, np.zeros_like(intermediate_final_mask)

    final_largest_cc_label = np.argmax(final_bincount[1:]) + 1
    final_largest_cc_mask = (final_labels == final_largest_cc_label).astype(np.uint8) * 255
    
    # Dilate this largest component
    dilated_final_cc = cv2.dilate(final_largest_cc_mask, kernel, iterations=body_dilation_iterations)
    
    # --- Step 9: Final, Final XOR operation ---
    final_mask = cv2.bitwise_xor(dilated_final_cc, full_body_mask_bin)

    return cleaned_mask, final_mask

def main():
    """
    Main function to parse arguments and process directories of masks.
    """
    parser = argparse.ArgumentParser(description="Process hair and body masks in batch.")
    parser.add_argument("--hair_mask_dir", required=True, help="Directory containing hair masks.")
    parser.add_argument("--body_mask_dir", required=True, help="Directory containing full-body masks.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the final processed masks.")
    parser.add_argument("--beard_mask_dir", default=None, help="Optional directory containing beard masks.")
    parser.add_argument("--beard_dilation", type=int, default=0, help="Optional kernel size to dilate the beard mask. No dilation if 0.")
    parser.add_argument("--body_dilation_iterations", type=int, default=1, help="Number of iterations for the body mask dilation.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get the list of hair masks to process
    hair_mask_files = [f for f in os.listdir(args.hair_mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not hair_mask_files:
        print(f"No image files found in {args.hair_mask_dir}. Exiting.")
        return

    print(f"Found {len(hair_mask_files)} masks to process.")

    for filename in hair_mask_files:
        hair_path = os.path.join(args.hair_mask_dir, filename)
        body_path = os.path.join(args.body_mask_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # Check if the corresponding full-body mask exists
        if not os.path.exists(body_path):
            print(f"Warning: No matching body mask found for {filename} at {body_path}. Skipping.")
            continue
        
        # Determine the path for the optional beard mask
        beard_path = None
        if args.beard_mask_dir:
            beard_path = os.path.join(args.beard_mask_dir, filename)

        # Process the masks
        _, final_result = clean_mask_with_hair(
            hair_path, 
            body_path,
            beard_mask_path=beard_path,
            beard_dilation_kernel_size=args.beard_dilation,
            body_dilation_iterations=args.body_dilation_iterations
        )

        # Save the final mask if it was processed successfully
        if final_result is not None:
            cv2.imwrite(output_path, final_result)
            print(f"Processed and saved: {filename}")

    print("\nBatch processing complete.")


if __name__ == '__main__':
    main()
