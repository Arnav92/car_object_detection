#!/usr/bin/env python3
"""
extra.py

Creates:
- data/extra.pdf  (histograms + example image + precision vs recall plot)
- (other helper functions included)

Usage:
    python extra.py
"""
import os
import glob
import math
import warnings
from typing import Dict, Optional

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
EXTRA_PDF = os.path.join(DATA_DIR, "extra.pdf")

SPEED_METRICS = os.path.join("speed_metrics.json")

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

IMGSZ = 640

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

def normalize_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


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


# ----------------- New: Precision vs Recall plot across models -----------------
def plot_precision_vs_recall(pdf: PdfPages):
    """
    Creates a Precision vs Recall plot with one line per model in MODEL_CONFIGS.
    Looks for results.csv for each model run and reads metrics/precision(B) and metrics/recall(B).
    Saves the figure to the provided PdfPages object.
    """
    # gather per-model series
    model_series = {}
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        run_name = cfg.get("run_name", "")
        path = find_results_csv_for_run(run_name)
        if not path or not os.path.exists(path):
            print(f"[PR] results.csv not found for {name} (run '{run_name}'). Skipping.")
            continue
        df = robust_read_results_csv(path)
        # ensure metric columns present
        pcol = "metrics/precision(B)"
        rcol = "metrics/recall(B)"
        if pcol not in df.columns or rcol not in df.columns:
            print(f"[PR] precision/recall columns missing in {path} for {name}. Skipping.")
            continue
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
        df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
        sub = df.dropna(subset=[pcol, rcol, "epoch"]).copy()
        if sub.empty:
            print(f"[PR] no valid precision/recall rows for {name} in {path}. Skipping.")
            continue
        model_series[name] = sub.sort_values("epoch")[["epoch", pcol, rcol]]

    if not model_series:
        print("[PR] No model precision/recall data available to plot.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("tab10")
    for i, (name, s) in enumerate(model_series.items()):
        recall = s["metrics/recall(B)"].values if "metrics/recall(B)" in s.columns else s.iloc[:, 2].values
        precision = s["metrics/precision(B)"].values if "metrics/precision(B)" in s.columns else s.iloc[:, 1].values
        # Plot with lines and markers
        ax.plot(recall, precision, marker="o", linestyle="-", label=name, color=cmap(i % 10))
        # annotate epoch numbers lightly
        for ep, rc, pr in zip(s["epoch"].values, recall, precision):
            ax.text(rc, pr, str(int(ep)), fontsize=6, alpha=0.6, va="bottom", ha="right")

    ax.set_title("Precision vs Recall (per-epoch points labeled by epoch)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.relim()
    ax.autoscale()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"[PR] Precision-vs-Recall page added to PDF.")


# ----------------- Histogram + example image (modified to accept PdfPages) -----------------
def plot_histograms_and_example_image(pdf: Optional[PdfPages] = None, output_pdf: str = EXTRA_PDF):
    """
    If pdf is provided, pages are written to it. Otherwise a new PDF at output_pdf is created.
    """
    own_pdf = False
    if pdf is None:
        pdf = PdfPages(output_pdf)
        own_pdf = True

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"{TRAIN_CSV} not found.")
    df = pd.read_csv(TRAIN_CSV)

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

    if own_pdf:
        pdf.close()
        print(f"Saved histograms + example image to {output_pdf}")
    else:
        print("Added histograms + example image pages to provided PdfPages object.")


# ----------------- Renaming helper -----------------
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


def plot_latency_histograms(pdf: PdfPages, speed_metrics_path: str = SPEED_METRICS):
    """
    Read speed_metrics.json and create an overlaid histogram of validation latencies (ms)
    for each model found in the file. Writes one page to the provided PdfPages object.
    """
    import json

    if not os.path.exists(speed_metrics_path):
        print(f"[LAT] speed metrics file not found: {speed_metrics_path}. Skipping latency histogram.")
        return

    with open(speed_metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # gather val.latencies_ms for each model key present
    model_latencies = {}
    for model_name, model_data in data.items():
        if model_name == "UnpretrainedYOLO":
            print("UnpretrainedYOLO skipped!")
            continue
        val = model_data.get("val", {})
        lat = val.get("latencies_ms") or val.get("latencies") or val.get("latency_ms")
        if lat and isinstance(lat, list) and len(lat) > 0:
            # coerce to floats
            try:
                lat_f = [float(x) for x in lat]
            except Exception:
                lat_f = []
            if lat_f:
                model_latencies[model_name] = np.array(lat_f)

    if not model_latencies:
        print(f"[LAT] No validation latency arrays found in {speed_metrics_path}.")
        return

    # determine common binning across all models (Freedman–Diaconis rule or fallback)
    all_vals = np.concatenate(list(model_latencies.values()))
    q25, q75 = np.percentile(all_vals, [25, 75])
    iqr = max(q75 - q25, 1e-6)
    bin_width = 2 * iqr * (len(all_vals) ** (-1/3)) if len(all_vals) > 0 else 1.0
    if bin_width <= 0 or np.isnan(bin_width):
        bins = 30
    else:
        bins = max(10, int(np.ceil((all_vals.max() - all_vals.min()) / bin_width)))
    # safety cap for bins
    bins = min(bins, 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, (name, lat_arr) in enumerate(sorted(model_latencies.items())):
        ax.hist(lat_arr, bins=bins, alpha=0.45, label=f"{name} (n={len(lat_arr)})", density=False,
                edgecolor="black", linewidth=0.3, color=cmap(i % 10))
        med = float(np.median(lat_arr))
        ax.axvline(med, color=cmap(i % 10), linestyle="--", linewidth=1.5)
        ax.text(med, ax.get_ylim()[1] * (0.85 - 0.08 * (i % 4)), f"med={med:.1f} ms", rotation=90,
                va="center", ha="right", fontsize=9, color=cmap(i % 10))

    ax.set_title("Validation Latency Histograms (ms) — per-model")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    # focus x-axis tightly around observed data (small padding)
    xmin = float(all_vals.min())
    xmax = float(all_vals.max())
    padding = max((xmax - xmin) * 0.03, 1.0)
    ax.set_xlim(max(0.0, xmin - padding), xmax + padding)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"[LAT] Latency histogram page added to PDF (models: {', '.join(model_latencies.keys())}).")


def plot_pr_at_iou(pdf: PdfPages, iou_thresh: float = 0.5, imgsz: int = IMGSZ, max_images: int = None):
    """
    Compute and plot Precision-Recall curves at a given IoU threshold for all models in MODEL_CONFIGS.
    Instead of labeling curves with the IoU, this function computes the area under each PR curve
    (approximate Average Precision) and shows it in the legend (AP ≈ area).
    Results are written as one page to the provided PdfPages object.

    Notes:
      - Expects workspace layout:
          workspace/val/images/*.jpg
          workspace/val/labels/*.txt   (YOLO normalized labels: class x_center y_center w h)
        and model weights at:
          workspace/runs/<run_name>/weights/*.pt
      - Uses greedy per-image per-class matching: highest-IoU unmatched GT >= iou_thresh counts as TP.
    """
    import os
    import glob
    from ultralytics import YOLO
    import cv2
    import numpy as np

    def find_best_weight_for_run(run_name: str) -> str:
        wdir = os.path.join(WORKSPACE, "runs", run_name, "weights")
        if not os.path.isdir(wdir):
            return ""
        for candidate in ("best.pt", "last.pt"):
            p = os.path.join(wdir, candidate)
            if os.path.exists(p):
                return p
        pts = glob.glob(os.path.join(wdir, "*.pt"))
        return pts[0] if pts else ""

    def read_yolo_labels_for_image(label_path: str, img_w: int, img_h: int):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                x1 = x_center - w / 2.0
                y1 = y_center - h / 2.0
                x2 = x_center + w / 2.0
                y2 = y_center + h / 2.0
                boxes.append((cls, max(0.0, x1), max(0.0, y1), min(img_w, x2), min(img_h, y2)))
        return boxes

    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    # --- load GTs ---
    val_images_dir = os.path.join(WORKSPACE, "val", "images")
    val_labels_dir = os.path.join(WORKSPACE, "val", "labels")
    if not os.path.isdir(val_images_dir) or not os.path.isdir(val_labels_dir):
        print(f"[PR-IOU] Expected val images/labels under {os.path.join(WORKSPACE, 'val')}; missing folder. Skipping PR.")
        return

    image_paths = sorted(glob.glob(os.path.join(val_images_dir, "*.*")))
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    if len(image_paths) == 0:
        print(f"[PR-IOU] No images found in {val_images_dir}; skipping.")
        return

    gt_by_image = {}
    total_gt = 0
    for p in image_paths:
        img_name = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(val_labels_dir, os.path.splitext(img_name)[0] + ".txt")
        boxes = read_yolo_labels_for_image(label_path, w, h)
        gt_by_image[img_name] = boxes
        total_gt += len(boxes)

    if total_gt == 0:
        print("[PR-IOU] No ground-truth boxes found in val labels; cannot compute PR.")
        return

    # --- run predictions for each model ---
    preds_by_model = {}
    for cfg in MODEL_CONFIGS:
        name = cfg.get("name", cfg.get("run_name", "model"))
        run = cfg.get("run_name", "")
        wpath = find_best_weight_for_run(run)
        if not wpath:
            print(f"[PR-IOU] No weights found for run '{run}' (model {name}); skipping.")
            continue
        print(f"[PR-IOU] Running predictions for {name} using weights: {wpath}")
        model = YOLO(wpath)
        model_preds = []
        for p in image_paths:
            img_name = os.path.basename(p)
            try:
                res = model.predict(source=p, imgsz=imgsz, verbose=False)[0]
            except Exception as e:
                print(f"[PR-IOU] predict failed for {p} with model {name}: {e}")
                continue
            if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.ones(len(xyxy))
            clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else np.zeros(len(xyxy))
            for (b, conf, cls) in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = [float(v) for v in b]
                model_preds.append({"image": img_name, "cls": int(cls), "conf": float(conf), "box": (x1, y1, x2, y2)})
        if len(model_preds) == 0:
            print(f"[PR-IOU] No predictions for {name}; skipping.")
            continue
        preds_by_model[name] = model_preds

    if not preds_by_model:
        print("[PR-IOU] No model predictions collected; skipping PR plotting.")
        return

    # --- compute PR (and AP area) per model ---
    pr_results = {}  # name -> (precisions, recalls, ap_area)
    for name, preds in preds_by_model.items():
        preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
        matched_gt = {img: [False] * len(gt_by_image.get(img, [])) for img in gt_by_image.keys()}

        tp = np.zeros(len(preds_sorted), dtype=np.int32)
        fp = np.zeros(len(preds_sorted), dtype=np.int32)

        for i, pred in enumerate(preds_sorted):
            img = pred["image"]
            pcls = pred["cls"]
            pbox = pred["box"]
            gts = gt_by_image.get(img, [])
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                gcls, gx1, gy1, gx2, gy2 = gt
                if int(gcls) != int(pcls):
                    continue
                if matched_gt.get(img) is None:
                    continue
                if matched_gt[img][j]:
                    continue
                iouv = iou_xyxy(pbox, (gx1, gy1, gx2, gy2))
                if iouv > best_iou:
                    best_iou = iouv
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1
                matched_gt[img][best_j] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp).astype(np.float32)
        cum_fp = np.cumsum(fp).astype(np.float32)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-12)
        recalls = cum_tp / float(total_gt)

        # ensure monotonic recall (non-decreasing) and proper integration bounds
        # prepend (0,1) style points commonly used for AP calc
        rec_for_area = np.concatenate(([0.0], recalls, [1.0]))
        prec_for_area = np.concatenate(([1.0], precisions, [0.0]))
        # optional: make precision envelope (monotonic decreasing) for classic AP -- we compute simple trapezoid
        ap_area = float(np.trapz(prec_for_area, rec_for_area))

        pr_results[name] = (precisions, recalls, ap_area)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("tab10")
    for i, (name, (prec, rec, ap)) in enumerate(pr_results.items()):
        if len(prec) == 0 or len(rec) == 0:
            continue
        label = f"{name} (AP≈{ap:.3f})"
        # plot stepwise PR curve
        ax.step(rec, prec, where='post', label=label, color=cmap(i % 10))
        # markers
        idxs = np.linspace(0, len(rec) - 1, min(12, max(1, len(rec)))).astype(int)
        ax.plot(rec[idxs], prec[idxs], 'o', markersize=3, color=cmap(i % 10))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision — Recall curves at IoU = {iou_thresh:.2f} (AP = area under PR curve)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    print(f"[PR-IOU] PR curves at IoU={iou_thresh:.2f} added to PDF (models: {', '.join(pr_results.keys())}).")


def main():
    # simple main that produces histograms/example image (existing function) then the PR page
    with PdfPages(EXTRA_PDF) as pdf:
        # existing histogram + example image page(s)
        plot_histograms_and_example_image(pdf=pdf)

        # latency histogram page
        plot_latency_histograms(pdf=pdf)

        # add PR curves page at chosen IoU
        IOU = 0.75  # change to desired IoU threshold
        plot_pr_at_iou(pdf=pdf, iou_thresh=IOU, imgsz=640, max_images=None)

    print(f"Saved combined PDF to {EXTRA_PDF}")




if __name__ == "__main__":
    main()