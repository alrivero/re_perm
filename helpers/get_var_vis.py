#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Tuple, List

def find_npy_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.npy") if p.is_file()])
    return sorted([p for p in root.glob("*.npy") if p.is_file()])

def load_variances(files: List[Path], allow_negative: bool=False) -> np.ndarray:
    vals = []
    for p in files:
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"[warn] Could not load {p}: {e}")
            continue
        arr = np.asarray(arr, dtype=np.float64).ravel()
        mask = np.isfinite(arr)
        arr = arr[mask]
        if not allow_negative:
            arr = arr[arr >= 0]
        if arr.size:
            vals.append(arr)
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals, axis=0)

def minmax_normalize(x: np.ndarray, vmin: float=None, vmax: float=None) -> Tuple[np.ndarray, float, float]:
    if x.size == 0:
        return x, np.nan, np.nan
    if vmin is None:
        vmin = float(np.nanmin(x))
    if vmax is None:
        vmax = float(np.nanmax(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x), vmin, vmax
    y = (x - vmin) / (vmax - vmin)
    return np.clip(y, 0.0, 1.0), vmin, vmax

def percentile_normalize(x: np.ndarray, low: float, high: float) -> Tuple[np.ndarray, float, float]:
    if x.size == 0:
        return x, np.nan, np.nan
    vmin = float(np.percentile(x, low))
    vmax = float(np.percentile(x, high))
    return minmax_normalize(x, vmin=vmin, vmax=vmax)

def soft_normalize(x: np.ndarray, scale: float=None) -> Tuple[np.ndarray, float, float]:
    """
    v_hat = x / (x + scale). If scale is None, use median positive variance.
    Returns (v_hat, dummy_vmin=0.0, used_scale).
    """
    if x.size == 0:
        return x, np.nan, np.nan
    pos = x[x > 0]
    if scale is None:
        scale = float(np.median(pos)) if pos.size else 1.0
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    vhat = x / (x + scale)
    return np.clip(vhat, 0.0, 1.0), 0.0, scale

def plot_hist(values: np.ndarray, bins: int, title: str, out_png: Path):
    if values.size == 0:
        print(f"[warn] No data to plot: {out_png.name}")
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(values, bins=bins, range=(0.0, 1.0))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_hist_csv(values: np.ndarray, bins: int, out_csv: Path):
    """Save histogram bin edges and counts for reproducibility."""
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left","bin_right","count"])
        for i in range(len(counts)):
            w.writerow([edges[i], edges[i+1], int(counts[i])])

def write_stats_csv(out_csv: Path, raw_count: int, vmin: float, vmax: float,
                    method: str, params: dict, var_norm: np.ndarray, conf: np.ndarray):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["method", method])
        for k,v in params.items():
            w.writerow([k, v])
        w.writerow(["raw_count", raw_count])
        w.writerow(["vmin_used", vmin])
        w.writerow(["vmax_or_scale_used", vmax])
        if var_norm.size:
            w.writerow(["var_norm_mean", float(np.mean(var_norm))])
            w.writerow(["var_norm_std", float(np.std(var_norm))])
            w.writerow(["var_norm_min", float(np.min(var_norm))])
            w.writerow(["var_norm_max", float(np.max(var_norm))])
        if conf.size:
            w.writerow(["conf_mean", float(np.mean(conf))])
            w.writerow(["conf_std", float(np.std(conf))])
            w.writerow(["conf_min", float(np.min(conf))])
            w.writerow(["conf_max", float(np.max(conf))])

def normalize_with_params(x: np.ndarray, method: str, vmin: float, vmax: float, scale: float):
    if method == "minmax":
        return np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(x)
    elif method == "percentile":
        return np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(x)
    else:  # soft
        return np.clip(x / (x + scale), 0.0, 1.0)

def main():
    ap = argparse.ArgumentParser(
        description="Normalize variance maps (.npy) to confidence and plot histograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-i", "--input-dir", required=True, type=Path,
                    help="Directory containing .npy variance maps.")
    ap.add_argument("-o", "--out-dir", default=Path("./variance_confidence_out"), type=Path,
                    help="Directory to write outputs (PNGs/CSV/NPY).")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories.")
    ap.add_argument("--bins", type=int, default=100,
                    help="Number of histogram bins (on [0,1]).")
    ap.add_argument("--method", choices=["minmax","percentile","soft"], default="minmax",
                    help="Normalization method for variance → [0,1].")
    ap.add_argument("--p-low", type=float, default=1.0,
                    help="Lower percentile for 'percentile' method.")
    ap.add_argument("--p-high", type=float, default=99.0,
                    help="Upper percentile for 'percentile' method.")
    ap.add_argument("--soft-scale", type=float, default=None,
                    help="Scale for 'soft' method; if omitted, uses median positive variance.")
    ap.add_argument("--save-arrays", action="store_true",
                    help="Save normalized variance and confidence arrays to .npy (global concatenated).")

    # NEW: per-map outputs
    ap.add_argument("--per-map", action="store_true",
                    help="Also write per-map histograms (normalized variance and confidence).")
    ap.add_argument("--per-map-dir", type=Path, default=None,
                    help="Directory for per-map histograms (default: OUT_DIR/per_map_hists).")
    ap.add_argument("--per-map-csv", action="store_true",
                    help="Additionally write per-map histogram counts to CSV.")

    ap.add_argument("--allow-negative", action="store_true",
                    help="Keep negative values if present (otherwise they are dropped).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_map_dir = args.per_map_dir or (args.out_dir / "per_map_hists")

    files = find_npy_files(args.input_dir, args.recursive)
    if not files:
        print(f"[error] No .npy files found in {args.input_dir} (recursive={args.recursive}).")
        return

    print(f"[info] Found {len(files)} files. Loading variances for global normalization…")
    v_all = load_variances(files, allow_negative=args.allow_negative)
    if v_all.size == 0:
        print("[error] No finite variance values found after cleaning.")
        return
    print(f"[info] Loaded {v_all.size} variance values (all files combined).")

    # Fit global normalization
    params = {}
    method = args.method
    if method == "minmax":
        vhat_all, vmin, vmax = minmax_normalize(v_all)
        params = {"vmin": vmin, "vmax": vmax}
        scale = np.nan
    elif method == "percentile":
        vhat_all, vmin, vmax = percentile_normalize(v_all, args.p_low, args.p_high)
        params = {"p_low": args.p_low, "p_high": args.p_high, "vmin(p)": vmin, "vmax(p)": vmax}
        scale = np.nan
    else:  # soft
        vhat_all, _, scale = soft_normalize(v_all, scale=args.soft_scale)
        params = {"scale": scale}
        vmin = 0.0
        vmax = scale  # for reporting only

    conf_all = 1.0 - vhat_all
    conf_all = np.clip(conf_all, 0.0, 1.0)

    # Global histograms
    plot_hist(vhat_all, args.bins, f"Normalized Variance (method={method})", args.out_dir / "hist_variance_normalized.png")
    plot_hist(conf_all, args.bins, f"Confidence = 1 - NormVariance (method={method})", args.out_dir / "hist_confidence.png")

    # Global stats CSV
    write_stats_csv(args.out_dir / "summary_stats.csv",
                    raw_count=v_all.size, vmin=vmin, vmax=vmax,
                    method=method, params=params,
                    var_norm=vhat_all, conf=conf_all)

    # Optional saves of concatenated arrays
    if args.save_arrays:
        np.save(args.out_dir / "variance_normalized.npy", vhat_all)
        np.save(args.out_dir / "confidence.npy", conf_all)

    # Per-map histograms (using the SAME global normalization params)
    if args.per_map:
        print(f"[info] Writing per-map histograms to {per_map_dir}")
        for p in files:
            try:
                arr = np.load(p)
            except Exception as e:
                print(f"[warn] Skipping {p}: {e}")
                continue
            x = np.asarray(arr, dtype=np.float64).ravel()
            x = x[np.isfinite(x)]
            if not args.allow_negative:
                x = x[x >= 0]
            if x.size == 0:
                print(f"[warn] {p.name}: no valid values after cleaning.")
                continue

            vhat = normalize_with_params(x, method=method, vmin=vmin, vmax=vmax, scale=scale)
            conf = np.clip(1.0 - vhat, 0.0, 1.0)

            stem = p.stem
            out_var = per_map_dir / f"{stem}__hist_var_norm.png"
            out_conf = per_map_dir / f"{stem}__hist_conf.png"
            plot_hist(vhat, args.bins, f"{stem} — NormVariance (method={method})", out_var)
            plot_hist(conf, args.bins, f"{stem} — Confidence (method={method})", out_conf)

            if args.per_map_csv:
                save_hist_csv(vhat, args.bins, per_map_dir / f"{stem}__hist_var_norm.csv")
                save_hist_csv(conf, args.bins, per_map_dir / f"{stem}__hist_conf.csv")

    print("\n[done] Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()