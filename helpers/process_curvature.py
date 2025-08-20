import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import os
import argparse
from tqdm import tqdm

def get_curvature_map(orientation_field, curve_length=65, mask=None, processing_mask=None):
    """
    Computes the curvature map.
    - Iterates only over pixels in the 'processing_mask' (eroded safety margin).
    - Checks path boundaries against the original 'mask'.
    - Sets curvature to 0 if a path is incomplete.
    """
    height, width = orientation_field.shape
    curvature_map = np.zeros_like(orientation_field)

    if mask is None:
        mask = np.ones_like(orientation_field, dtype=np.uint8)
    if processing_mask is None:
        processing_mask = mask

    y_coords, x_coords = np.arange(height), np.arange(width)
    interp_cos = RectBivariateSpline(y_coords, x_coords, np.cos(2 * orientation_field))
    interp_sin = RectBivariateSpline(y_coords, x_coords, np.sin(2 * orientation_field))

    q = (curve_length - 1) // 2

    rows, cols = np.where(processing_mask > 0)
    pixel_iterator = zip(rows, cols)

    for r, c in pixel_iterator:
        # --- Forward Tracing ---
        fwd_x, fwd_y = float(c), float(r)
        fwd_steps_taken = 0
        for i in range(q):
            curr_x, curr_y = int(round(fwd_x)), int(round(fwd_y))
            if not (0 <= curr_y < height and 0 <= curr_x < width and mask[curr_y, curr_x] > 0):
                break
            
            angle = 0.5 * np.arctan2(interp_sin(fwd_y, fwd_x, grid=False), interp_cos(fwd_y, fwd_x, grid=False))
            fwd_x += np.cos(angle)
            fwd_y += np.sin(angle)
            fwd_steps_taken = i + 1

        # --- Trace Backward Path ---
        bwd_x, bwd_y = float(c), float(r)
        bwd_steps_taken = 0
        for i in range(q):
            curr_x, curr_y = int(round(bwd_x)), int(round(bwd_y))
            if not (0 <= curr_y < height and 0 <= curr_x < width and mask[curr_y, curr_x] > 0):
                break

            angle = 0.5 * np.arctan2(interp_sin(bwd_y, bwd_x, grid=False), interp_cos(bwd_y, bwd_x, grid=False))
            bwd_x -= np.cos(angle)
            bwd_y -= np.sin(angle)
            bwd_steps_taken = i + 1
            
        # --- Check if path was completed ---
        if fwd_steps_taken < q or bwd_steps_taken < q:
            curvature_map[r, c] = 0
        else:
            fwd_x, fwd_y = np.clip(fwd_x, 0, width-1), np.clip(fwd_y, 0, height-1)
            bwd_x, bwd_y = np.clip(bwd_x, 0, width-1), np.clip(bwd_y, 0, height-1)

            orient_start = 0.5 * np.arctan2(interp_sin(bwd_y, bwd_x, grid=False), interp_cos(bwd_y, bwd_x, grid=False))
            orient_mid = 0.5 * np.arctan2(interp_sin(r, c, grid=False), interp_cos(r, c, grid=False))
            orient_end = 0.5 * np.arctan2(interp_sin(fwd_y, fwd_x, grid=False), interp_cos(fwd_y, fwd_x, grid=False))
            
            diff1 = min(np.abs(orient_mid - orient_start), np.pi - np.abs(orient_mid - orient_start))
            diff2 = min(np.abs(orient_end - orient_mid), np.pi - np.abs(orient_end - orient_mid))

            curvature_map[r, c] = diff1 + diff2
            
    return curvature_map

def main(args):
    os.makedirs(args.curve_dir, exist_ok=True)
    os.makedirs(args.vis_img_dir, exist_ok=True)

    map_files = sorted([f for f in os.listdir(args.orient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(map_files)} pre-computed orientation maps to process.")

    for map_name in tqdm(map_files, desc="Processing Curvature"):
        basename = os.path.splitext(map_name)[0]

        orient_map_path = os.path.join(args.orient_dir, map_name)
        orient_map_rgb = cv2.imread(orient_map_path)
        
        if orient_map_rgb is None:
            print(f"\nWarning: Could not read orientation map {map_name}. Skipping.")
            continue
            
        orient_map_rgb = orient_map_rgb.astype(np.float32) / 255.0
        
        b, g, r = orient_map_rgb[..., 0], orient_map_rgb[..., 1], orient_map_rgb[..., 2]
        
        dx = b * 2.0 - 1.0
        dy = g * 2.0 - 1.0
        mask = (r > 0.5).astype(np.uint8)
        
        orientation = np.arctan2(dy, dx)
        
        # Create a processing mask with a safety margin
        kernel_size = 2 * args.edge_margin + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processing_mask = cv2.erode(mask, kernel, iterations=1)

        # Calculate curvature using the safety margin
        curvature = get_curvature_map(orientation, args.curve_length, mask=mask, processing_mask=processing_mask)
        
        np.save(os.path.join(args.curve_dir, f'{basename}.npy'), curvature.astype(np.float16))

        # Visualization
        curvature_normalized = cv2.normalize(curvature, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        curvature_heatmap = cv2.applyColorMap(curvature_normalized, cv2.COLORMAP_INFERNO)
        
        # FIX: Use the eroded 'processing_mask' to black out the background AND the halo.
        curvature_heatmap[processing_mask == 0] = 0
        
        cv2.imwrite(os.path.join(args.vis_img_dir, f'{basename}_curvature.png'), curvature_heatmap)
        
    print(f"\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate curvature maps from pre-computed RGB orientation maps.')
    
    parser.add_argument('--img_path', required=True, type=str, help='Input directory for HIGH-RESOLUTION source images.')
    parser.add_argument('--orient_dir', required=True, type=str, help='Input directory for RGB-encoded orientation maps.')
    
    parser.add_argument('--curve_dir', required=True, type=str, help='Output directory for the final curvature maps (.npy).')
    parser.add_argument('--vis_img_dir', required=True, type=str, help='Output directory for visualizations (.png).')
    
    parser.add_argument('--curve_length', default=55, type=int, help='The pixel length of the curve used to calculate curvature.')
    parser.add_argument('--edge_margin', default=8, type=int, help='Number of pixels to ignore at the mask boundary to prevent artifacts.')
    
    args = parser.parse_args()
    main(args)