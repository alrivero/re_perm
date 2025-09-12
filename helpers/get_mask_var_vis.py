#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv

def find_images(root: Path, recursive: bool):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if recursive:
        return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return sorted([p for p in root.glob("*") if p.suffix.lower() in exts])

def load_pixels(files):
    vals = []
    for f in files:
        try:
            img = Image.open(f).convert("L")  # grayscale
            arr = np.array(img, dtype=np.float32).ravel() / 255.0
            # Exclude 0 and 1 values
            arr = arr[(arr > 0.0) & (arr < 1.0)]
            vals.append(arr)
        except Exception as e:
            print(f"[warn] Skipping {f}: {e}")
    return np.concatenate(vals) if vals else np.array([])

def plot_hist(values, bins, title, out_png):
    if values.size == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(values, bins=bins, range=(0,1))
    plt.title(title)
    plt.xlabel("Normalized pixel value [0,1] (excl. 0 and 1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_hist_csv(values, bins, out_csv):
    counts, edges = np.histogram(values, bins=bins, range=(0,1))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left","bin_right","count"])
        for i in range(len(counts)):
            w.writerow([edges[i], edges[i+1], int(counts[i])])

def main():
    ap = argparse.ArgumentParser(
        description="Analyze image mattes (0–255), normalize to [0,1], exclude 0s and 1s, and plot histograms."
    )
    ap.add_argument("-i", "--input-dir", required=True, type=Path,
                    help="Directory containing image mattes.")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("./matte_hist_out"),
                    help="Directory to save outputs.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    ap.add_argument("--bins", type=int, default=100, help="Number of histogram bins.")
    ap.add_argument("--per-map", action="store_true",
                    help="Also save per-image histograms.")
    ap.add_argument("--per-map-csv", action="store_true",
                    help="Save per-image histogram CSVs.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_map_dir = args.out_dir / "per_map_hists"

    files = find_images(args.input_dir, args.recursive)
    if not files:
        print(f"[error] No images found in {args.input_dir}")
        return

    print(f"[info] Found {len(files)} images. Loading pixel values…")
    all_pixels = load_pixels(files)
    if all_pixels.size == 0:
        print("[error] No valid pixel data found after excluding 0s and 1s.")
        return

    # Global histogram
    plot_hist(all_pixels, args.bins, "Global Matte Histogram (excl. 0s and 1s)", args.out_dir / "hist_global.png")
    save_hist_csv(all_pixels, args.bins, args.out_dir / "hist_global.csv")

    # Per-map histograms
    if args.per_map:
        print(f"[info] Writing per-map histograms to {per_map_dir}")
        for f in files:
            try:
                img = Image.open(f).convert("L")
                arr = np.array(img, dtype=np.float32).ravel() / 255.0
                arr = arr[(arr > 0.0) & (arr < 1.0)]
            except Exception as e:
                print(f"[warn] Skipping {f}: {e}")
                continue

            if arr.size == 0:
                print(f"[warn] {f.name}: all pixels were 0 or 1.")
                continue

            stem = f.stem
            out_png = per_map_dir / f"{stem}_hist.png"
            plot_hist(arr, args.bins, f"Histogram for {stem} (excl. 0s and 1s)", out_png)

            if args.per_map_csv:
                out_csv = per_map_dir / f"{stem}_hist.csv"
                save_hist_csv(arr, args.bins, out_csv)

    print(f"[done] Outputs saved to {args.out_dir}")

if __name__ == "__main__":
    main()