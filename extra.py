#!/usr/bin/env python3
"""
extra.py

Regression analysis on metric *differences* between model pairs (YOLO, FastYOLO, UnpretrainedYOLO).
Generates a PDF with a plot per (pair, metric) showing epoch vs difference and the fitted linear
regression line; annotates slope, intercept, p-values, R^2, and a short conclusion about
statistical significance — with annotations placed below the plot (no overlap).

Also contains histogram + example single-box image code.

Outputs:
- data/regression_report.pdf
- data/regression_summary.csv
- data/data_description_histograms.pdf

Usage:
    python extra.py
"""
import os
import glob
import math
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")

# Try to import scipy.stats for p-values and t-distribution functions
try:
    from scipy import stats as _scipy_stats
    SCIPY_STATS = True
except Exception:
    _scipy_stats = None
    SCIPY_STATS = False

# ---------- User paths & configs ----------
DATA_DIR = os.path.join("data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_solution_bounding_boxes (1).csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
PERSONAL_IMAGES_DIR = os.path.join(DATA_DIR, "personal_images")

WORKSPACE = "workspace"
MODEL_CONFIGS = [
    {"name": "YOLO", "run_name": "yolo_s_acc"},
    {"name": "FastYOLO", "run_name": "yolo_n_fast"},
    {"name": "UnpretrainedYOLO", "run_name": "untrained_yolo_s_acc"}
]

# Where to write outputs
REGRESSION_PDF = os.path.join(DATA_DIR, "regression_report.pdf")
REGRESSION_SUMMARY_CSV = os.path.join(DATA_DIR, "regression_summary.csv")
HISTOGRAMS_PDF = os.path.join(DATA_DIR, "data_description_histograms.pdf")

# Candidate metric columns to attempt (ordered preference)
CANDIDATE_METRICS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "speed/val_median_ms",
    "speed/val_p95_ms",
    "speed/val_mean_ms",
    "speed/val_fps",
    "speed/val_throughput_fps",
    "speed/val_flops_g"
]


# ----------------- Helpers -----------------
def find_results_csv_for_run(run_name: str) -> str:
    """
    Attempt to find results.csv for a run name under workspace/runs/detect/<run_name>/results.csv
    """
    if not run_name:
        return ""
    candidate = os.path.join(WORKSPACE, "runs", "detect", run_name, "results.csv")
    if os.path.exists(candidate):
        return candidate
    alt = os.path.join(WORKSPACE, "runs", run_name, "results.csv")
    if os.path.exists(alt):
        return alt
    # Search recursively
    pattern = os.path.join(WORKSPACE, "runs", "**", run_name, "results.csv")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    # any results.csv under runs that contains the run_name substring
    matches2 = [p for p in glob.glob(os.path.join(WORKSPACE, "runs", "**", "results.csv"), recursive=True)
                if run_name in p]
    if matches2:
        return matches2[0]
    return ""


def robust_read_results_csv(path: str) -> pd.DataFrame:
    """Read results.csv and ensure an 'epoch' column exists (best-effort)."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback to python engine to be robust
        df = pd.read_csv(path, engine="python", error_bad_lines=False, warn_bad_lines=False)
    if "epoch" not in df.columns:
        # try to infer epoch column from the first column if it looks numeric
        first_col = df.columns[0]
        try:
            maybe = pd.to_numeric(df[first_col], errors="coerce")
            if maybe.notna().sum() > 0:
                # insert inferred epoch (fill forward for any missing)
                df.insert(0, "epoch", maybe.fillna(method="ffill").astype(int))
            else:
                df.insert(0, "epoch", np.arange(len(df)))
        except Exception:
            df.insert(0, "epoch", np.arange(len(df)))
    # coerce to numeric and drop rows without epoch
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def linear_regression_stats(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compute simple linear regression (y = intercept + slope * x) statistics via OLS (closed form).
    Returns dictionary with slope (b1), intercept (b0), se_slope, se_intercept, t_slope, p_slope, t_intercept, p_intercept, r2, n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return {"n": n, "slope": None, "intercept": None, "se_slope": None, "se_intercept": None,
                "t_slope": None, "p_slope": None, "t_intercept": None, "p_intercept": None, "r2": None}
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = np.sum((x - x_mean) ** 2)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    slope = float(Sxy / Sxx) if Sxx != 0 else 0.0
    intercept = float(y_mean - slope * x_mean)
    # predictions & residuals
    y_pred = intercept + slope * x
    resid = y - y_pred
    SSE = np.sum(resid ** 2)
    dof = max(n - 2, 1)
    s2 = SSE / dof if dof > 0 else np.nan
    # standard errors
    se_slope = float(math.sqrt(s2 / Sxx)) if (Sxx != 0 and s2 >= 0) else None
    se_intercept = float(math.sqrt(s2 * (1.0 / n + (x_mean ** 2) / Sxx))) if (Sxx != 0 and s2 >= 0) else None
    # t-stats
    t_slope = float(slope / se_slope) if (se_slope and se_slope != 0) else None
    t_intercept = float(intercept / se_intercept) if (se_intercept and se_intercept != 0) else None
    # p-values via scipy if available
    p_slope = None
    p_intercept = None
    if SCIPY_STATS:
        try:
            p_slope = float(2.0 * _scipy_stats.t.sf(abs(t_slope), df=dof)) if t_slope is not None else None
            p_intercept = float(2.0 * _scipy_stats.t.sf(abs(t_intercept), df=dof)) if t_intercept is not None else None
        except Exception:
            p_slope = None
            p_intercept = None
    # R^2
    SS_tot = np.sum((y - y_mean) ** 2)
    r2 = float(1.0 - SSE / SS_tot) if (SS_tot != 0) else 0.0
    return {
        "n": int(n),
        "slope": slope,
        "intercept": intercept,
        "se_slope": se_slope,
        "se_intercept": se_intercept,
        "t_slope": t_slope,
        "p_slope": p_slope,
        "t_intercept": t_intercept,
        "p_intercept": p_intercept,
        "r2": r2
    }


def format_val(v, fmt: str = ".6f"):
    """Safely format numeric values or return 'N/A' for None/NaN."""
    if v is None:
        return "N/A"
    try:
        if isinstance(v, (int, np.integer)):
            return f"{v:d}"
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return "N/A"
            return format(v, fmt)
        return str(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return "N/A"


# ----------------- Regression report -----------------
def run_regression_analysis_and_report(output_pdf: str = REGRESSION_PDF,
                                       summary_csv: str = REGRESSION_SUMMARY_CSV):
    """
    Orchestrates regression analysis and PDF summary.
    """
    # 1) Find results.csv for each model
    model_results = {}
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        run_name = cfg.get("run_name", "")
        p = find_results_csv_for_run(run_name)
        if p:
            print(f"Found results.csv for {name}: {p}")
        else:
            print(f"Warning: results.csv not found for run '{run_name}' (model {name}).")
        model_results[name] = p

    # 2) Load dataframes
    dfs = {}
    for name, p in model_results.items():
        dfs[name] = robust_read_results_csv(p) if p else pd.DataFrame()

    # 3) Determine metrics present
    all_cols = set()
    for df in dfs.values():
        if not df.empty:
            all_cols.update(df.columns.tolist())
    # choose metrics: intersection with candidate list
    metrics_to_use = [m for m in CANDIDATE_METRICS if m in all_cols]
    if not metrics_to_use:
        # fallback to any metrics/ columns
        metrics_to_use = sorted([c for c in all_cols if c.startswith("metrics/")])
    if not metrics_to_use:
        print("No metrics found in results.csv files to analyze. Exiting regression step.")
        return

    print("Metrics to analyze:", metrics_to_use)

    pairs = [
        ("YOLO", "FastYOLO"),
        ("YOLO", "UnpretrainedYOLO"),
        ("FastYOLO", "UnpretrainedYOLO")
    ]

    summary_records = []

    # Start PDF
    with PdfPages(output_pdf) as pdf:
        for (A, B) in pairs:
            dfA = dfs.get(A, pd.DataFrame()).copy()
            dfB = dfs.get(B, pd.DataFrame()).copy()
            if dfA.empty or dfB.empty:
                print(f"Skipping pair {A} vs {B} because one or both result CSVs are missing.")
                continue

            # perform inner join on epoch
            merged = pd.merge(dfA, dfB, on="epoch", suffixes=(f".{A}", f".{B}"))
            if merged.empty:
                print(f"No overlapping epochs found for {A} vs {B}; skipping.")
                continue

            for metric in metrics_to_use:
                colA = metric + f".{A}"
                colB = metric + f".{B}"
                if colA not in merged.columns or colB not in merged.columns:
                    # skip if missing in merged
                    continue

                # coerce numeric and drop NA
                merged[colA] = pd.to_numeric(merged[colA], errors="coerce")
                merged[colB] = pd.to_numeric(merged[colB], errors="coerce")
                sub = merged.dropna(subset=[colA, colB, "epoch"]).copy()
                if sub.empty or len(sub) < 2:
                    # not enough points to regress
                    print(f"Not enough data for {A} vs {B} on metric {metric}; need >=2 rows.")
                    continue

                sub["diff"] = sub[colA] - sub[colB]
                x = sub["epoch"].values
                y = sub["diff"].values

                stats = linear_regression_stats(x, y)
                alpha = 0.05
                slope_sig = (stats["p_slope"] is not None and stats["p_slope"] < alpha)
                intercept_sig = (stats["p_intercept"] is not None and stats["p_intercept"] < alpha)

                # Add to summary
                summary_records.append({
                    "pair": f"{A}_vs_{B}",
                    "metric": metric,
                    "n": stats["n"],
                    "slope": stats["slope"],
                    "se_slope": stats["se_slope"],
                    "t_slope": stats["t_slope"],
                    "p_slope": stats["p_slope"],
                    "slope_significant": slope_sig,
                    "intercept": stats["intercept"],
                    "se_intercept": stats["se_intercept"],
                    "t_intercept": stats["t_intercept"],
                    "p_intercept": stats["p_intercept"],
                    "intercept_significant": intercept_sig,
                    "r2": stats["r2"]
                })

                # -----------------------------
                # Create a page with plot on top and text below
                # -----------------------------
                fig = plt.figure(figsize=(10, 9))
                gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
                ax = fig.add_subplot(gs[0])
                ax_text = fig.add_subplot(gs[1])

                # Plot scatter and regression line on ax
                ax.scatter(x, y, alpha=0.8, label="Observed diff (A - B)")
                if stats["slope"] is not None:
                    xs = np.linspace(x.min(), x.max(), 200)
                    ys = stats["intercept"] + stats["slope"] * xs
                    ax.plot(xs, ys, label="Fit (linear)", color="red")
                ax.axhline(0, color="gray", linestyle="--", linewidth=1)
                ax.set_title(f"{A} − {B} : {metric}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(f"Difference in {metric} (A − B)")
                ax.grid(True)
                ax.legend(loc="upper left")

                # Build annotation text block (safe formatting)
                text_lines = [
                    f"n = {format_val(stats['n'], '')}",
                    f"slope = {format_val(stats['slope'])}    (se = {format_val(stats['se_slope'])})",
                    f"t(slope) = {format_val(stats['t_slope'], '.3f')}    p(slope) = {format_val(stats['p_slope'], '.6f')}",
                    f"intercept = {format_val(stats['intercept'])}    (se = {format_val(stats['se_intercept'])})",
                    f"t(intercept) = {format_val(stats['t_intercept'], '.3f')}    p(intercept) = {format_val(stats['p_intercept'], '.6f')}",
                    f"R² = {format_val(stats['r2'], '.4f')}"
                ]

                interp_lines = []
                if stats["p_slope"] is not None:
                    if slope_sig:
                        interp_lines.append("Slope ≠ 0 (statistically significant) → metric difference changes with epoch.")
                    else:
                        interp_lines.append("Slope ≈ 0 (not significant) → metric difference not correlated with epoch.")
                else:
                    interp_lines.append("Slope p-value unavailable; cannot determine significance here.")

                if stats["p_intercept"] is not None:
                    if intercept_sig:
                        interp_lines.append("Intercept is significant → persistent baseline difference between models.")
                    else:
                        interp_lines.append("Intercept not significant → no consistent baseline difference detected.")
                else:
                    interp_lines.append("Intercept p-value unavailable; cannot determine significance here.")

                full_text = "\n".join(text_lines + [""] + interp_lines)

                # Put text in ax_text (below plot). Use monospace for alignment.
                ax_text.axis("off")
                # Place text at the top-left of the lower panel, allow newlines.
                ax_text.text(0.01, 0.98, full_text, fontsize=9, va="top", ha="left", family="monospace")

                # Save page
                pdf.savefig(fig)
                plt.close(fig)

    # Write summary CSV
    if summary_records:
        try:
            summary_df = pd.DataFrame(summary_records)
            summary_df.to_csv(summary_csv, index=False)
            print(f"Wrote regression summary CSV: {summary_csv}")
        except Exception as e:
            print(f"Could not write regression summary CSV: {e}")
    else:
        print("No summary records were produced (no regressions ran).")

    print(f"Wrote regression report PDF: {output_pdf}")


# ----------------- Histogram + example image -----------------
def plot_histograms_and_example_image(output_pdf: str = HISTOGRAMS_PDF):
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"{TRAIN_CSV} not found.")
    df = pd.read_csv(TRAIN_CSV)

    pdf = PdfPages(output_pdf)

    # 1) bounding box areas (normalize by image area)
    image_w = 676
    image_h = 380
    image_area = float(image_w * image_h)
    df["width"] = pd.to_numeric(df["xmax"], errors="coerce") - pd.to_numeric(df["xmin"], errors="coerce")
    df["height"] = pd.to_numeric(df["ymax"], errors="coerce") - pd.to_numeric(df["ymin"], errors="coerce")
    df["area"] = (df["width"] * df["height"]) / image_area
    avg_area = float(df["area"].dropna().mean())

    plt.figure(figsize=(8, 5))
    plt.hist(df["area"].dropna(), bins=50, edgecolor="black")
    plt.title(f"Histogram of Bounding Box Areas (Average Area = {avg_area:.3f})")
    plt.xlabel("Bounding Box Area (fraction of image area)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig()
    plt.close()

    # 2) histogram of object count per image (only images with >=1 box in current code)
    image_box_count_series = df.groupby("image").size()
    avg_boxes = float(image_box_count_series.mean())

    plt.figure(figsize=(8, 5))
    plt.hist(image_box_count_series, bins=range(0, int(image_box_count_series.max()) + 2),
             align="left", edgecolor="black", rwidth=0.85)
    plt.title(f"Histogram of Object Count per Image (Average = {avg_boxes:.2f})")
    plt.xlabel("Number of Cars per Image")
    plt.ylabel("Count of Images")
    plt.grid(True, linestyle="--", alpha=0.6)
    pdf.savefig()
    plt.close()

    # 3) Example single-box image whose box area is closest to avg_area
    one_box_counts = df.groupby("image").size()
    one_box_images = one_box_counts[one_box_counts == 1].index.tolist()

    if len(one_box_images) > 0:
        single_df = df[df["image"].isin(one_box_images)].copy()
        single_df["area"] = ((single_df["xmax"] - single_df["xmin"]) * (single_df["ymax"] - single_df["ymin"])) / image_area
        single_df["area_diff"] = (single_df["area"] - avg_area).abs()
        # pick best row
        selected = single_df.sort_values("area_diff").iloc[0]
        example_image = selected["image"]
        img_path = os.path.join(TRAIN_IMAGES_DIR, example_image)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            ax = plt.gca()
            rect = Rectangle(
                (selected["xmin"], selected["ymin"]),
                selected["xmax"] - selected["xmin"],
                selected["ymax"] - selected["ymin"],
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            plt.title(
                f"1-Car Example Near Avg Area\n"
                f"{example_image} | box area={format_val(selected['area'], '.3f')}, "
                f"avg={format_val(avg_area, '.3f')}, Δ={format_val(selected['area_diff'], '.3f')}"
            )
            plt.axis("off")
            pdf.savefig()
            plt.close()
        else:
            print(f"Example image {img_path} not found; skipping example image page.")
    else:
        print("No single-box images found; skipping example image page.")

    pdf.close()
    print(f"Saved histograms + example image to {output_pdf}")


def rename_personal_files():
    """Renaming all files in ./data/personal_images"""
    # List all files in the folder
    files = [f for f in os.listdir(PERSONAL_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Sort files alphabetically so numbering is consistent
    files.sort()

    # Rename each file
    for idx, filename in enumerate(files, start=1):
        # Always save as .jpeg
        new_name = f"{idx}.jpeg"
        old_path = os.path.join(PERSONAL_IMAGES_DIR, filename)
        new_path = os.path.join(PERSONAL_IMAGES_DIR, new_name)

        # Rename (overwrite if duplicate exists)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} → {new_name}")

    print("✅ Renaming completed!")


def main():
    print("Generating histograms + example image ...")
    try:
        plot_histograms_and_example_image(HISTOGRAMS_PDF)
    except Exception as e:
        print(f"Histogram generation failed: {e}")

    print("Running regression analysis and generating PDF ...")
    try:
        run_regression_analysis_and_report(REGRESSION_PDF, REGRESSION_SUMMARY_CSV)
    except Exception as e:
        print(f"Regression analysis failed: {e}")


if __name__ == "__main__":
    rename_personal_files()
