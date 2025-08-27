#!/usr/bin/env python3
"""
extra.py

Regression analysis on metric *differences* between model pairs (YOLO, FastYOLO, UnpretrainedYOLO).
Generates a PDF with a page per (pair, metric) showing:

  - LEFT: epoch vs metric for both models (points + per-model linear fit)
  - RIGHT: epoch vs (A - B) difference (points + linear fit)
  - BOTTOM: textual stats (slope/intercept/p-values/R^2/etc.) placed below the two plots
    so text never overlaps the plots.

Also contains histogram + example single-box image code and a helper to rename personal images.

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
from typing import Dict

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
    # convert to int safely
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
        # return all keys with None-friendly values
        return {
            "n": int(n), "slope": None, "intercept": None, "se_slope": None, "se_intercept": None,
            "t_slope": None, "p_slope": None, "t_intercept": None, "p_intercept": None, "r2": None
        }
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
    Orchestrates regression analysis and PDF summary with LEFT=per-model plots and RIGHT=difference plots,
    and BOTTOM=text area containing stats (no overlap).
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
                # columns in merged will be exactly metric names (no suffix) if original dfs had same column names;
                # but we merged so need to refer to original metric names. If metric isn't present with suffix we try original names:
                if colA not in merged.columns and metric in merged.columns:
                    colA = metric
                    colB = metric  # both refer to same column names - but merged contains duplicates; adjust:
                    # If this fallback happens (rare), rename columns from A/B dfs before merging would be better.
                if colA not in merged.columns or colB not in merged.columns:
                    # skip if missing in merged
                    # Usually merged will contain e.g. "metrics/mAP50(B).YOLO" style; earlier code sets suffixes.
                    # If metric missing, continue.
                    continue

                # coerce numeric and drop NA
                merged[colA] = pd.to_numeric(merged[colA], errors="coerce")
                merged[colB] = pd.to_numeric(merged[colB], errors="coerce")
                sub = merged.dropna(subset=[colA, colB, "epoch"]).copy()
                if sub.empty or len(sub) < 2:
                    # not enough points to regress
                    print(f"Not enough data for {A} vs {B} on metric {metric}; need >=2 rows.")
                    continue

                # prepare per-model series for left plot
                A_df = sub[["epoch", colA]].dropna().copy()
                B_df = sub[["epoch", colB]].dropna().copy()

                # compute per-model regression stats
                stats_A = linear_regression_stats(A_df["epoch"].values, A_df[colA].values) if len(A_df) >= 2 else {
                    "n": int(len(A_df)), "slope": None, "intercept": None, "se_slope": None, "se_intercept": None,
                    "t_slope": None, "p_slope": None, "t_intercept": None, "p_intercept": None, "r2": None
                }
                stats_B = linear_regression_stats(B_df["epoch"].values, B_df[colB].values) if len(B_df) >= 2 else {
                    "n": int(len(B_df)), "slope": None, "intercept": None, "se_slope": None, "se_intercept": None,
                    "t_slope": None, "p_slope": None, "t_intercept": None, "p_intercept": None, "r2": None
                }

                # diff
                sub["diff"] = sub[colA] - sub[colB]
                x = sub["epoch"].values
                y = sub["diff"].values
                stats_diff = linear_regression_stats(x, y)
                alpha = 0.05
                slope_sig = (stats_diff["p_slope"] is not None and stats_diff["p_slope"] < alpha)
                intercept_sig = (stats_diff["p_intercept"] is not None and stats_diff["p_intercept"] < alpha)

                # Add to summary (store per-pair-diff stats)
                summary_records.append({
                    "pair": f"{A}_vs_{B}",
                    "metric": metric,
                    "n": stats_diff["n"],
                    "slope": stats_diff["slope"],
                    "se_slope": stats_diff["se_slope"],
                    "t_slope": stats_diff["t_slope"],
                    "p_slope": stats_diff["p_slope"],
                    "slope_significant": slope_sig,
                    "intercept": stats_diff["intercept"],
                    "se_intercept": stats_diff["se_intercept"],
                    "t_intercept": stats_diff["t_intercept"],
                    "p_intercept": stats_diff["p_intercept"],
                    "intercept_significant": intercept_sig,
                    "r2": stats_diff["r2"]
                })

                # -----------------------------
                # Create a page: top row has LEFT and RIGHT plots; bottom row has text.
                # -----------------------------
                fig = plt.figure(figsize=(11.7, 9))  # A4-ish landscape
                # make bottom row a bit taller to ensure text fits cleanly
                gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], width_ratios=[1, 1], hspace=0.3, wspace=0.35)
                ax_left = fig.add_subplot(gs[0, 0])
                ax_right = fig.add_subplot(gs[0, 1])
                ax_text = fig.add_subplot(gs[1, :])

                # LEFT: both models points & individual linear fits
                colors = ("tab:blue", "tab:orange")
                # plot points
                ax_left.scatter(A_df["epoch"], A_df[colA], alpha=0.9, label=f"{A} (points)", marker="o", color=colors[0])
                ax_left.scatter(B_df["epoch"], B_df[colB], alpha=0.9, label=f"{B} (points)", marker="s", color=colors[1])
                # per-model fit lines (if available)
                if stats_A.get("slope") is not None and stats_A.get("intercept") is not None:
                    xsA = np.linspace(A_df["epoch"].min(), A_df["epoch"].max(), 200)
                    ysA = stats_A["intercept"] + stats_A["slope"] * xsA
                    ax_left.plot(xsA, ysA, color=colors[0], linestyle="--", label=f"{A} fit")
                if stats_B.get("slope") is not None and stats_B.get("intercept") is not None:
                    xsB = np.linspace(B_df["epoch"].min(), B_df["epoch"].max(), 200)
                    ysB = stats_B["intercept"] + stats_B["slope"] * xsB
                    ax_left.plot(xsB, ysB, color=colors[1], linestyle="--", label=f"{B} fit")
                ax_left.set_title(f"{metric} — {A} & {B}")
                ax_left.set_xlabel("Epoch")
                ax_left.set_ylabel(metric)
                ax_left.grid(True)
                ax_left.legend(loc="best", fontsize=9)

                # RIGHT: difference plot and fit
                ax_right.scatter(x, y, alpha=0.85, label="Observed diff (A - B)", color="tab:green")
                if stats_diff.get("slope") is not None and stats_diff.get("intercept") is not None:
                    xs = np.linspace(x.min(), x.max(), 200)
                    ys = stats_diff["intercept"] + stats_diff["slope"] * xs
                    ax_right.plot(xs, ys, label="Fit (diff)", color="red")
                ax_right.axhline(0, color="gray", linestyle="--", linewidth=1)
                ax_right.set_title(f"{A} − {B} : difference")
                ax_right.set_xlabel("Epoch")
                ax_right.set_ylabel(f"Difference ({metric})")
                ax_right.grid(True)
                ax_right.legend(loc="best", fontsize=9)

                # BOTTOM: textual block. include both per-model stats and diff stats, with intercept details
                ax_text.axis("off")
                lines = []
                lines.append(f"PAIR: {A}  vs  {B}     METRIC: {metric}")
                lines.append("-" * 110)
                # A stats (include intercept & its stats)
                lines.append(
                    f"{A}: n={format_val(stats_A.get('n'), '')}   slope={format_val(stats_A.get('slope'))} (se={format_val(stats_A.get('se_slope'))})   "
                    f"t={format_val(stats_A.get('t_slope'), '.3f')}   p={format_val(stats_A.get('p_slope'), '.6f')}   R²={format_val(stats_A.get('r2'), '.4f')}"
                )
                lines.append(
                    f"    intercept={format_val(stats_A.get('intercept'))} (se={format_val(stats_A.get('se_intercept'))})   "
                    f"t={format_val(stats_A.get('t_intercept'), '.3f')}   p={format_val(stats_A.get('p_intercept'), '.6f')}"
                )
                # B stats (include intercept & its stats)
                lines.append(
                    f"{B}: n={format_val(stats_B.get('n'), '')}   slope={format_val(stats_B.get('slope'))} (se={format_val(stats_B.get('se_slope'))})   "
                    f"t={format_val(stats_B.get('t_slope'), '.3f')}   p={format_val(stats_B.get('p_slope'), '.6f')}   R²={format_val(stats_B.get('r2'), '.4f')}"
                )
                lines.append(
                    f"    intercept={format_val(stats_B.get('intercept'))} (se={format_val(stats_B.get('se_intercept'))})   "
                    f"t={format_val(stats_B.get('t_intercept'), '.3f')}   p={format_val(stats_B.get('p_intercept'), '.6f')}"
                )
                lines.append("-" * 110)
                # diff stats
                lines.append(
                    f"DIFF (A − B): n={format_val(stats_diff.get('n'), '')}   slope={format_val(stats_diff.get('slope'))} (se={format_val(stats_diff.get('se_slope'))})   "
                    f"t={format_val(stats_diff.get('t_slope'), '.3f')}   p={format_val(stats_diff.get('p_slope'), '.6f')}   R²={format_val(stats_diff.get('r2'), '.4f')}"
                )
                lines.append(
                    f"    intercept = {format_val(stats_diff.get('intercept'))} (se={format_val(stats_diff.get('se_intercept'))})   "
                    f"t={format_val(stats_diff.get('t_intercept'), '.3f')}   p={format_val(stats_diff.get('p_intercept'), '.6f')}"
                )
                lines.append("")
                # Interpretation short
                if stats_diff.get("p_slope") is not None:
                    if stats_diff["p_slope"] < 0.05:
                        lines.append("Interpretation: slope is statistically significant → difference changes with epoch.")
                    else:
                        lines.append("Interpretation: slope not significant → difference does not systematically change with epoch.")
                else:
                    lines.append("Interpretation: slope p-value unavailable in this environment.")
                if stats_diff.get("p_intercept") is not None:
                    if stats_diff["p_intercept"] < 0.05:
                        lines.append("Baseline: intercept significant → persistent baseline difference between models across epochs.")
                    else:
                        lines.append("Baseline: intercept not significant → no consistent baseline difference detected.")
                else:
                    lines.append("Baseline: intercept p-value unavailable in this environment.")
                # Render text
                text_block = "\n".join(lines)
                ax_text.text(0.01, 0.98, text_block, fontsize=9, va="top", ha="left", family="monospace")

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
    """Renaming all files in ./data/personal_images to 1.jpeg, 2.jpeg, ..."""
    if not os.path.exists(PERSONAL_IMAGES_DIR):
        print(f"Personal images dir not found: {PERSONAL_IMAGES_DIR}")
        return
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
        # If destination already exists, attempt to remove it first to avoid OSError
        try:
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} → {new_name}")
        except Exception as e:
            print(f"Could NOT rename {filename} → {new_name}: {e}")
    print("Renaming completed!")


def main():
    print("Running regression analysis and generating PDF ...")
    try:
        run_regression_analysis_and_report(REGRESSION_PDF, REGRESSION_SUMMARY_CSV)
    except Exception as e:
        print(f"Regression analysis failed: {e}")


if __name__ == "__main__":
    main()
