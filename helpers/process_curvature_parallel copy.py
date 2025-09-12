import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import os
import tqdm
import cv2
import argparse

def get_sobel_kernels(device):
    """Creates Sobel filters for gradient calculation as Conv2d layers."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    sobel_x_layer = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    sobel_y_layer = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    
    sobel_x_layer.weight.data = sobel_x
    sobel_y_layer.weight.data = sobel_y
    
    return sobel_x_layer, sobel_y_layer

def get_gaussian_blur_layer(sigma, kernel_size, device=None):
    """Creates a Gaussian blur filter as a Conv2d layer."""
    # Create a 1D Gaussian kernel
    k = cv2.getGaussianKernel(kernel_size, sigma)
    # Create a 2D kernel from the 1D kernel
    k2d = k @ k.transpose()
    # Convert to a PyTorch tensor
    kernel_tensor = torch.from_numpy(k2d).float().unsqueeze(0).unsqueeze(0).to(device)

    blur_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
    blur_layer.weight.data = kernel_tensor
    blur_layer.weight.requires_grad = False
    return blur_layer

def calculate_curvature_gpu(orientation_map, mask_tensor, device, confidence_map_tensor=None, confidence_integration='multiply', blur_sigma=5, blur_kernel_size=21):
    """
    Calculates the curvature map from an orientation field on the GPU.
    """
    sobel_x, sobel_y = get_sobel_kernels(device)

    # 1. Convert orientation angles to a continuous double-angle vector field
    cos_2theta = torch.cos(2 * orientation_map)
    sin_2theta = torch.sin(2 * orientation_map)

    # Apply mask to the vector field components
    cos_2theta = cos_2theta * mask_tensor
    sin_2theta = sin_2theta * mask_tensor

    # --- ADVANCED: Confidence-Weighted Smoothing ---
    if confidence_map_tensor is not None and confidence_integration == 'smoothing':
        blur_layer = get_gaussian_blur_layer(sigma=blur_sigma, kernel_size=blur_kernel_size, device=device)
        
        # Create a blurred, "safe" version of the orientation field
        with torch.no_grad():
            cos_2theta_blurred = blur_layer(cos_2theta.unsqueeze(0).unsqueeze(0)).squeeze()
            sin_2theta_blurred = blur_layer(sin_2theta.unsqueeze(0).unsqueeze(0)).squeeze()

        # Blend the original and blurred fields based on confidence
        # Where confidence is high, use original. Where low, use blurred.
        confidence = confidence_map_tensor 
        cos_2theta = (cos_2theta * confidence) + (cos_2theta_blurred * (1 - confidence))
        sin_2theta = (sin_2theta * confidence) + (sin_2theta_blurred * (1 - confidence))

    # 2. Calculate gradients of the (potentially smoothed) vector field
    with torch.no_grad():
        d_cos_dx = sobel_x(cos_2theta.unsqueeze(0).unsqueeze(0)).squeeze()
        d_cos_dy = sobel_y(cos_2theta.unsqueeze(0).unsqueeze(0)).squeeze()
        d_sin_dx = sobel_x(sin_2theta.unsqueeze(0).unsqueeze(0)).squeeze()
        d_sin_dy = sobel_y(sin_2theta.unsqueeze(0).unsqueeze(0)).squeeze()

    # 3. Estimate curvature as the magnitude of the gradients
    curvature = torch.sqrt(d_cos_dx**2 + d_cos_dy**2 + d_sin_dx**2 + d_sin_dy**2)
    
    # --- SIMPLE: Post-hoc Multiplication ---
    if confidence_map_tensor is not None and confidence_integration == 'multiply':
        curvature = curvature * confidence_map_tensor

    # Apply mask again to the final output and move to CPU
    curvature = curvature * mask_tensor
    return curvature.cpu().numpy()

def main(args):
    # --- Validation for blur kernel size ---
    if args.blur_kernel_size % 2 == 0:
        raise ValueError("--blur_kernel_size must be an odd number.")

    # --- Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. This script requires a GPU.")
        return

    # --- Create Output Directories ---
    os.makedirs(args.curve_dir, exist_ok=True)
    os.makedirs(args.vis_img_dir, exist_ok=True)
    if args.save_average_map:
        avg_map_dir = os.path.dirname(args.save_average_map)
        if avg_map_dir:
            os.makedirs(avg_map_dir, exist_ok=True)
    if args.save_average_scalar:
        avg_scalar_dir = os.path.dirname(args.save_average_scalar)
        if avg_scalar_dir:
            os.makedirs(avg_scalar_dir, exist_ok=True)


    # --- Auto-calculate variance range if needed ---
    if args.use_confidence and (args.var_min is None or args.var_max is None):
        print("Variance range not provided. Calculating from all variance maps...")
        var_files = [f for f in os.listdir(args.confidence_dir) if f.endswith('.npy')]
        if not var_files:
            raise FileNotFoundError(f"No .npy files found in confidence directory: {args.confidence_dir}")

        global_min = np.inf
        global_max = -np.inf
        for var_name in tqdm.tqdm(var_files, desc="Scanning variance maps"):
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

    map_files = sorted([f for f in os.listdir(args.orient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
    
    print(f"Found {len(map_files)} orientation maps.")
    print(f"Orientation Format: {args.orient_format}")

    # --- Initialize for Averaging ---
    total_curvature_sum = None
    total_pixel_count = 0
    image_count = 0

    for map_name in tqdm.tqdm(map_files, desc="Processing images"):
        basename = os.path.splitext(map_name)[0]
        
        try:
            # --- Load Mask and Orientation ---
            mask_path = os.path.join(args.mask_dir, f"{basename}.png")
            if not os.path.exists(mask_path):
                # print(f"Warning: Mask for {basename} not found. Skipping.")
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask).to(device)
            
            orient_map_path = os.path.join(args.orient_dir, map_name)
            
            if args.orient_format == 'angle':
                orient_map_gray = cv2.imread(orient_map_path, cv2.IMREAD_GRAYSCALE)
                if orient_map_gray is None: continue
                orientation_rad = orient_map_gray.astype(np.float32) * (np.pi / 180.0)
            else: # 'vector' format
                # This part is CPU-based but happens once per image
                orient_map_rgb = cv2.imread(orient_map_path, cv2.IMREAD_COLOR)
                if orient_map_rgb is None: continue
                orient_map_rgb = cv2.cvtColor(orient_map_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                channel_map = {'R': 0, 'G': 1, 'B': 2}
                dx = orient_map_rgb[:, :, channel_map[args.dx_channel]] * 2.0 - 1.0
                dy = orient_map_rgb[:, :, channel_map[args.dy_channel]] * 2.0 - 1.0
                orientation_rad = np.arctan2(dy, dx)
            
            orientation_tensor = torch.from_numpy(orientation_rad).to(device)

            # --- Load Confidence Map ---
            confidence_tensor = None
            if args.use_confidence:
                conf_path = os.path.join(args.confidence_dir, f'{basename}.npy')
                if os.path.exists(conf_path):
                    variance_map = np.load(conf_path)
                    # Convert variance to confidence [0, 1]
                    clipped_var = np.clip(variance_map, args.var_min, args.var_max)
                    norm_var = (clipped_var - args.var_min) / (args.var_max - args.var_min)
                    confidence_map = 1.0 - norm_var
                    confidence_tensor = torch.from_numpy(confidence_map).to(device)

            # --- Core GPU Calculation ---
            curvature = calculate_curvature_gpu(
                orientation_tensor, mask_tensor, device, 
                confidence_tensor, args.confidence_integration,
                args.blur_sigma, args.blur_kernel_size
            )

            # --- Accumulate for Averaging ---
            if args.save_average_map or args.save_average_scalar:
                if total_curvature_sum is None:
                    total_curvature_sum = np.zeros_like(curvature, dtype=np.float64)
                
                if total_curvature_sum.shape == curvature.shape:
                    total_curvature_sum += curvature
                    total_pixel_count += np.sum(mask > 0.1)
                    image_count += 1
                else:
                    print(f"Warning: Shape mismatch for {basename}. Skipping from average calculation.")

            # --- Save Outputs ---
            np.save(os.path.join(args.curve_dir, f'{basename}.npy'), curvature.astype(np.float16))

            # --- Create and Save Visualization ---
            vis_mask = (mask > 0.1).astype(np.uint8)

            if args.no_vis_normalization:
                # Clip to a reasonable range (e.g., 0 to pi, as curvature is sum of gradients)
                # and scale to 0-255. This preserves the absolute scale across images.
                # A max value of 4.0 is a reasonable guess for this gradient magnitude.
                scaled_curvature = np.clip(curvature, 0, 4.0) * (255.0 / 4.0)
                curvature_normalized = scaled_curvature.astype(np.uint8)
            else:
                # Default behavior: normalize each image to its own min/max
                curvature_normalized = cv2.normalize(curvature, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            curvature_heatmap = cv2.applyColorMap(curvature_normalized, cv2.COLORMAP_INFERNO)
            curvature_heatmap[vis_mask == 0] = 0
            cv2.imwrite(os.path.join(args.vis_img_dir, f'{basename}.png'), curvature_heatmap)
        
        except Exception as e:
            print(f"Failed to process {map_name}: {e}")

    # --- Finalize and Save Averages ---
    if image_count > 0:
        if args.save_average_map:
            print(f"\nCalculating and saving average map from {image_count} images...")
            average_curvature_map = total_curvature_sum / image_count
            np.save(args.save_average_map, average_curvature_map.astype(np.float16))
            print(f"Average map saved to {args.save_average_map}")

            # Save visualization of the average map
            avg_vis_path = os.path.splitext(args.save_average_map)[0] + '.png'
            avg_norm = cv2.normalize(average_curvature_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            avg_heatmap = cv2.applyColorMap(avg_norm, cv2.COLORMAP_INFERNO)
            cv2.imwrite(avg_vis_path, avg_heatmap)
            print(f"Average map visualization saved to {avg_vis_path}")

        if args.save_average_scalar:
            if total_pixel_count > 0:
                average_scalar = np.sum(total_curvature_sum) / total_pixel_count
                print(f"\nOverall Average Curvature Scalar: {average_scalar:.6f}")
                with open(args.save_average_scalar, 'w') as f:
                    f.write(str(average_scalar))
                print(f"Average scalar saved to {args.save_average_scalar}")

    print(f"\nProcessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-accelerated curvature estimation from orientation maps.')

    # --- I/O Arguments ---
    parser.add_argument('--orient_dir', required=True, type=str, help='Input directory for orientation maps.')
    parser.add_argument('--mask_dir', required=True, type=str, help="Input directory for masks.")
    parser.add_argument('--curve_dir', required=True, type=str, help='Output directory for curvature maps (.npy).')
    parser.add_argument('--vis_img_dir', required=True, type=str, help='Output directory for visualizations (.png).')
    
    # --- Format Arguments ---
    parser.add_argument('--orient_format', required=True, choices=['vector', 'angle'], 
                        help="Format of the orientation maps: 'vector' (dx/dy in channels) or 'angle' (pixel value is degrees).")
    parser.add_argument('--dx_channel', default='B', choices=['R', 'G', 'B'], help="Channel for dx component (used with 'vector' format).")
    parser.add_argument('--dy_channel', default='G', choices=['R', 'G', 'B'], help="Channel for dy component (used with 'vector' format).")

    # --- Confidence Arguments ---
    parser.add_argument('--use_confidence', action='store_true', help='Flag to enable confidence weighting.')
    parser.add_argument('--confidence_dir', type=str, help='Input directory for variance maps (.npy). Required if --use_confidence.')
    parser.add_argument('--var_min', type=float, help='(Optional) Variance value to map to max confidence. If not set, will be auto-calculated.')
    parser.add_argument('--var_max', type=float, help='(Optional) Variance value to map to min confidence. If not set, will be auto-calculated.')
    parser.add_argument('--confidence_integration', default='multiply', choices=['multiply', 'smoothing'], 
                        help="How to integrate confidence: 'multiply' (post-hoc) or 'smoothing' (pre-gradient).")

    # --- Smoothing Kernel Arguments ---
    parser.add_argument('--blur_sigma', type=float, default=30.0, help="Sigma for the Gaussian blur kernel used in 'smoothing' mode.")
    parser.add_argument('--blur_kernel_size', type=int, default=151, help="Size for the Gaussian blur kernel used in 'smoothing' mode. Must be an odd number.")

    # --- Averaging Arguments ---
    parser.add_argument('--save_average_map', type=str, help='(Optional) Filepath to save the final average curvature map (.npy).')
    parser.add_argument('--save_average_scalar', type=str, help='(Optional) Filepath to save the final average curvature value (.txt).')

    # --- Visualization Argument ---
    parser.add_argument('--no_vis_normalization', action='store_true',
                        help='Disables per-image normalization for visualizations to show absolute curvature scale.')

    args = parser.parse_args()
    main(args)

