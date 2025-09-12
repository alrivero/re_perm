import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import os
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def convert_variance_to_confidence(variance_map, var_min, var_max):
    """Converts a variance map to a confidence map."""
    # Clip values to the specified range
    clipped_variance = np.clip(variance_map, var_min, var_max)
    # Normalize the clipped values to a [0, 1] range
    # Add a small epsilon to prevent division by zero if var_min == var_max
    normalized_variance = (clipped_variance - var_min) / ((var_max - var_min) + 1e-9)
    # Invert the result so that low variance means high confidence
    confidence_map = 1.0 - normalized_variance
    return confidence_map

def process_pixel(r, c, height, width, interp_cos, interp_sin, interp_conf, interp_mask,
                  curve_length, mode, threshold):
    """
    Computes the curvature for a single pixel (r, c). This is the worker function for parallel processing.
    """
    q = (curve_length - 1) // 2
    path_confidences = []

    # Check that starting point is valid
    if interp_mask(r, c, grid=False).item() <= 0:
        return r, c, 0.0

    # --- Forward Tracing ---
    fwd_x, fwd_y = float(c), float(r)
    fwd_steps_taken = 0
    for i in range(q):
        # Use interpolated mask for sub-pixel boundary checks
        if not (0 <= fwd_y < height - 1 and 0 <= fwd_x < width - 1 and interp_mask(fwd_y, fwd_x, grid=False).item() > 0.1):
             break
        if interp_conf:
            confidence = interp_conf(fwd_y, fwd_x, grid=False).item()
            if mode == 'terminate' and confidence < threshold:
                break
            if mode == 'path':
                path_confidences.append(confidence)
        angle = 0.5 * np.arctan2(interp_sin(fwd_y, fwd_x, grid=False), interp_cos(fwd_y, fwd_x, grid=False))
        fwd_x += np.cos(angle)
        fwd_y += np.sin(angle)
        fwd_steps_taken = i + 1

    # --- Backward Tracing ---
    bwd_x, bwd_y = float(c), float(r)
    bwd_steps_taken = 0
    for i in range(q):
        if not (0 <= bwd_y < height - 1 and 0 <= bwd_x < width - 1 and interp_mask(bwd_y, bwd_x, grid=False).item() > 0.1):
             break
        if interp_conf:
            confidence = interp_conf(bwd_y, bwd_x, grid=False).item()
            if mode == 'terminate' and confidence < threshold:
                break
            if mode == 'path':
                path_confidences.append(confidence)
        angle = 0.5 * np.arctan2(interp_sin(bwd_y, bwd_x, grid=False), interp_cos(bwd_y, bwd_x, grid=False))
        bwd_x -= np.cos(angle)
        bwd_y -= np.sin(angle)
        bwd_steps_taken = i + 1

    if fwd_steps_taken < q or bwd_steps_taken < q:
        return r, c, 0.0

    fwd_x, fwd_y = np.clip(fwd_x, 0, width - 1), np.clip(fwd_y, 0, height - 1)
    bwd_x, bwd_y = np.clip(bwd_x, 0, width - 1), np.clip(bwd_y, 0, height - 1)

    orient_start = 0.5 * np.arctan2(interp_sin(bwd_y, bwd_x, grid=False), interp_cos(bwd_y, bwd_x, grid=False))
    orient_mid = 0.5 * np.arctan2(interp_sin(r, c, grid=False), interp_cos(r, c, grid=False))
    orient_end = 0.5 * np.arctan2(interp_sin(fwd_y, fwd_x, grid=False), interp_cos(fwd_y, fwd_x, grid=False))

    diff1 = min(np.abs(orient_mid - orient_start), np.pi - np.abs(orient_mid - orient_start))
    diff2 = min(np.abs(orient_end - orient_mid), np.pi - np.abs(orient_end - orient_mid))
    calculated_curvature = diff1 + diff2

    final_curvature = calculated_curvature
    if interp_conf and mode != 'none' and mode != 'terminate':
        if mode == 'keypoint':
            conf_start = interp_conf(bwd_y, bwd_x, grid=False).item()
            conf_mid = interp_conf(r, c, grid=False).item()
            conf_end = interp_conf(fwd_y, fwd_x, grid=False).item()
            weight = min(conf_start, conf_mid, conf_end)
            final_curvature *= weight
        elif mode == 'path' and path_confidences:
            path_confidences.append(interp_conf(r, c, grid=False).item())
            avg_confidence = sum(path_confidences) / len(path_confidences)
            final_curvature *= avg_confidence
    
    return r, c, final_curvature

def get_curvature_map(orientation_field, curve_length=65, mask=None,
                      confidence_map=None, mode='none', threshold=0.25, num_workers=-1, basename=''):
    """
    Computes the curvature map in parallel with a progress bar for pixels.
    """
    height, width = orientation_field.shape
    curvature_map = np.zeros_like(orientation_field, dtype=np.float32)

    if mask is None:
        mask = np.ones_like(orientation_field, dtype=np.float32)

    y_coords, x_coords = np.arange(height), np.arange(width)
    interp_cos = RectBivariateSpline(y_coords, x_coords, np.cos(2 * orientation_field))
    interp_sin = RectBivariateSpline(y_coords, x_coords, np.sin(2 * orientation_field))
    interp_mask = RectBivariateSpline(y_coords, x_coords, mask)
    
    interp_conf = None
    if confidence_map is not None and mode != 'none':
        interp_conf = RectBivariateSpline(y_coords, x_coords, confidence_map)

    rows, cols = np.where(mask > 0.1) # Use a small threshold for continuous masks
    
    # Create an iterator with a tqdm progress bar for pixels
    pixel_iterator = tqdm(zip(rows, cols), total=len(rows), desc=f"Pixels for {basename}", leave=False, unit='pix')
    
    results = Parallel(n_jobs=num_workers, backend="loky")(
        delayed(process_pixel)(
            r, c, height, width, interp_cos, interp_sin, interp_conf, interp_mask,
            curve_length, mode, threshold
        ) for r, c in pixel_iterator
    )

    # No need to print "Assembling..." as the user can see the pixel bar finish
    for r, c, curve_val in results:
        curvature_map[r, c] = curve_val
            
    return curvature_map

def main(args):
    # --- Argument Validation ---
    if args.orient_format == 'angle' and not args.mask_dir:
        raise ValueError("--mask_dir is required when using --orient_format 'angle'.")
    if args.confidence_mode != 'none' and not args.confidence_dir:
        raise ValueError("--confidence_dir is required when using a confidence mode.")
    
    # --- Auto-calculate variance range if not provided ---
    if args.confidence_mode != 'none' and (args.var_min is None or args.var_max is None):
        print("Variance range not provided. Calculating from all variance maps...")
        var_files = [f for f in os.listdir(args.confidence_dir) if f.endswith('.npy')]
        if not var_files:
            raise FileNotFoundError(f"No .npy files found in confidence directory: {args.confidence_dir}")

        global_min = np.inf
        global_max = -np.inf
        for var_name in tqdm(var_files, desc="Scanning variance maps"):
            var_path = os.path.join(args.confidence_dir, var_name)
            variance_map = np.load(var_path)
            global_min = min(global_min, np.min(variance_map))
            global_max = max(global_max, np.max(variance_map))
        
        args.var_min = global_min
        args.var_max = global_max
        print(f"Calculated variance range: min={args.var_min:.4f}, max={args.var_max:.4f}")
        if args.var_min >= args.var_max:
             print("Warning: Calculated min variance is not less than max. Confidence map may be uniform.")
             args.var_max = args.var_min + 1e-6 # Add epsilon to avoid division by zero

    os.makedirs(args.curve_dir, exist_ok=True)
    os.makedirs(args.vis_img_dir, exist_ok=True)

    channel_map = {'R': 0, 'G': 1, 'B': 2}
    map_files = sorted([f for f in os.listdir(args.orient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    print(f"Found {len(map_files)} orientation maps to process.")
    print(f"Orientation Format: {args.orient_format}")
    print(f"Confidence Mode: {args.confidence_mode}")
    print(f"Using {args.num_workers if args.num_workers > 0 else 'all available'} CPU cores.")
    if args.downscale_factor > 1.0:
        print(f"Downscaling maps by a factor of {args.downscale_factor}")
    
    # --- Automatic Curve Length Adjustment ---
    scaled_curve_length = int(round(args.curve_length / args.downscale_factor))
    # Ensure it's an odd number and at least 3
    if scaled_curve_length % 2 == 0:
        scaled_curve_length += 1
    scaled_curve_length = max(3, scaled_curve_length)
    if args.downscale_factor > 1.0:
        print(f"Original curve length {args.curve_length}px scaled to {scaled_curve_length}px for downscaled images.")

    # Main image loop with an outer progress bar
    for map_name in tqdm(map_files, desc="Overall Image Progress", unit='image'):
        basename = os.path.splitext(map_name)[0]

        # --- Load Maps ---
        mask = None
        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, f"{basename}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                continue
        
        if args.orient_format == 'angle':
            orient_map_path = os.path.join(args.orient_dir, map_name)
            orientation_map = cv2.imread(orient_map_path, cv2.IMREAD_GRAYSCALE)
            if orientation_map is None: continue
        else: # vector
            orient_map_path = os.path.join(args.orient_dir, map_name)
            orientation_map = cv2.imread(orient_map_path, cv2.IMREAD_COLOR)
            if orientation_map is None: continue

        confidence_map = None
        if args.confidence_mode != 'none':
            conf_path = os.path.join(args.confidence_dir, f'{basename}.npy')
            if os.path.exists(conf_path):
                confidence_map = np.load(conf_path).astype(np.float32) # FIX: Convert to float32 after loading

        # --- DOWNSCALING ---
        if args.downscale_factor > 1.0:
            h, w = orientation_map.shape[:2]
            new_h, new_w = int(h / args.downscale_factor), int(w / args.downscale_factor)
            
            orientation_map = cv2.resize(orientation_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if confidence_map is not None:
                confidence_map = cv2.resize(confidence_map, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # --- Process Orientation Format ---
        if args.orient_format == 'angle':
            orientation = orientation_map.astype(np.float32) * (np.pi / 180.0)
        else: # vector
            orient_map_rgb = cv2.cvtColor(orientation_map, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if mask is None:
                mask = (orient_map_rgb[:, :, channel_map['R']] > 0.5).astype(np.float32)
            dx = orient_map_rgb[:, :, channel_map[args.dx_channel]] * 2.0 - 1.0
            dy = orient_map_rgb[:, :, channel_map[args.dy_channel]] * 2.0 - 1.0
            orientation = np.arctan2(dy, dx)
        
        # --- Process Confidence ---
        if confidence_map is not None:
            confidence_map = convert_variance_to_confidence(confidence_map, args.var_min, args.var_max)

        # --- CORE CALCULATION ---
        curvature = get_curvature_map(
            orientation, scaled_curve_length, 
            mask=mask, 
            confidence_map=confidence_map, mode=args.confidence_mode, 
            threshold=args.conf_threshold, num_workers=args.num_workers,
            basename=basename
        )
        
        if mask is not None:
            curvature *= mask

        # --- Save Outputs ---
        np.save(os.path.join(args.curve_dir, f'{basename}.npy'), curvature.astype(np.float16))

        vis_mask = (mask > 0.1).astype(np.uint8) if mask is not None else np.ones_like(curvature, dtype=np.uint8)
        curvature_normalized = cv2.normalize(curvature, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        curvature_heatmap = cv2.applyColorMap(curvature_normalized, cv2.COLORMAP_INFERNO)
        curvature_heatmap[vis_mask == 0] = 0
        cv2.imwrite(os.path.join(args.vis_img_dir, f'{basename}.png'), curvature_heatmap)
        
    print(f"\nProcessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate curvature maps from orientation maps using CPU parallelization.')
    
    parser.add_argument('--orient_dir', required=True, type=str, help='Input directory for orientation maps.')
    parser.add_argument('--mask_dir', type=str, help="(Required for 'angle' format) Input directory for external masks.")
    parser.add_argument('--curve_dir', required=True, type=str, help='Output directory for curvature maps (.npy).')
    parser.add_argument('--vis_img_dir', required=True, type=str, help='Output directory for visualizations (.png).')
    
    parser.add_argument('--orient_format', required=True, choices=['vector', 'angle'], 
                        help="Format of the orientation maps: 'vector' (dx/dy in channels) or 'angle' (pixel value is degrees).")
    parser.add_argument('--dx_channel', default='B', choices=['R', 'G', 'B'], help="Channel for dx component (used with 'vector' format).")
    parser.add_argument('--dy_channel', default='G', choices=['R', 'G', 'B'], help="Channel for dy component (used with 'vector' format).")

    parser.add_argument('--curve_length', default=55, type=int, help='Pixel length of the curve used to calculate curvature (at native resolution).')
    
    parser.add_argument('--confidence_mode', default='none', choices=['none', 'keypoint', 'path', 'terminate'],
                        help='Method for incorporating confidence: "keypoint" weighting, "path" weighting, or "terminate" on low confidence.')
    parser.add_argument('--confidence_dir', type=str, help='Input directory for variance maps (.npy). Required for confidence modes.')
    parser.add_argument('--var_min', type=float, help='(Optional) Variance value to map to max confidence. If not set, will be auto-calculated.')
    parser.add_argument('--var_max', type=float, help='(Optional) Variance value to map to min confidence. If not set, will be auto-calculated.')
    parser.add_argument('--conf_threshold', default=0.25, type=float, help='Confidence threshold for "terminate" mode.')

    parser.add_argument('--num_workers', default=-1, type=int, 
                        help='Number of CPU cores to use for parallel processing. -1 means use all available cores.')
    
    # --- NEW DOWNSCALING ARGUMENT ---
    parser.add_argument('--downscale_factor', type=float, default=1.0, help='Factor by which to downscale input maps. 1.0 = no downscaling.')

    args = parser.parse_args()
    main(args)

