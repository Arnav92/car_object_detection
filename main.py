# What this script does:
# 1) Reads data/train_solution_bounding_boxes (1).csv
# 2) Creates a YOLOv8-style dataset folder under ./workspace/
# 3) Trains two models:
#    - "YOLO"  -> yolov8s (more accurate, slower)
#    - "FastYOLO" -> yolov8n (faster, smaller)
#    - "UnpretrainedYOLO" -> yolov8s.yaml (no pretraining)
# 4) Saves trained models to ./models/
# 5) Runs trained models on test images and writes test_predictions.csv
# 6) Generates a PDF report with loss curves, detection samples, AND speed graphs
#    (inference ms, FPS, FLOPs) for validation and test sets.
#
# Notes:
# - In the "main" function, the "train" function is commented out. Training is assumed
#   to have been done; if you want to re-train, uncomment it. The script will reuse
#   weights if found under ./models/ or workspace runs.
# - Speed measurements: warmup (default 10), per-image timing, report p50 & p95 latencies,
#   FPS computed from median latency. FLOPs estimated via Ultralytics model.info() or
#   via thop if available.
# ---------------------------------------------------------------

import os
import math
import glob
import time
import json
import shutil
import random
import warnings
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore", category=UserWarning) # Ignoring filterwarnings

from ultralytics import YOLO # "You only look once" pretrained model to use and train further on Kaggle dataset

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, KFold
import statistics
import re

# Defining plot style
font = {'weight': 'bold', 'size': 11}
matplotlib.rc('font', **font)

# -------------------------------
# User-adjustable variables
# -------------------------------
DATA_DIR = os.path.join("data")  # expects ./data/training_images, ./data/testing_images
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "testing_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train_solution_bounding_boxes (1).csv")
PERSONAL_DIR = os.path.join(DATA_DIR, "personal_images")

WORKSPACE = "workspace"  # we'll build YOLOv8 format dataset here
MODELS_DIR = "models"    # store/check pretrained + trained weights here
REPORT_PDF = "car_detection_report.pdf"
SPEED_JSON = "speed_metrics.json"

# Training settings
EPOCHS = 20
BATCH_SIZE = 16
IMGSZ = 640  # YOLO standard
VAL_SPLIT = 0.1 # 90% training and 10% validation choice
RANDOM_SEED = 42
USE_KFOLD = False
KFOLD_SPLITS = 5

# Two model configs
MODEL_CONFIGS = [
    {
        "name": "YOLO",           # accurate
        "pretrained": "yolov8s.pt",
        "run_name": "yolo_s_acc"
    },
    {
        "name": "FastYOLO",       # fast
        "pretrained": "yolov8n.pt",
        "run_name": "yolo_n_fast"
    },
    {
        "name": "UnpretrainedYOLO",       # no pretraining
        "pretrained": "yolov8s.yaml",
        "run_name": "untrained_yolo_s_acc"
    }
]

# Class names
CLASS_NAMES = ["car"]
NC = len(CLASS_NAMES)

# Define dataset paths (useful if not running training)
base_dir = "workspace/"
ds_paths = {
    "train_images": f"{base_dir}/train/images",
    "val_images": f"{base_dir}/val/images",
    "test_images": f"{base_dir}/test/images"
}

# Define pretrained weights (useful if not training and/or trained before)
per_model = {
    "YOLO": {
        "best_weights": "workspace/runs/yolo_s_acc/weights/best.pt",
        "results_csv": "workspace/runs/yolo_s_acc/results.csv"
    },
    "FastYOLO": {
        "best_weights": "workspace/runs/yolo_n_fast/weights/best.pt",
        "results_csv": "workspace/runs/yolo_n_fast/results.csv"
    },
    "UnpretrainedYOLO": {
        "best_weights": "workspace/runs/untrained_yolo_s_acc/weights/best.pt",
        "results_csv": "workspace/runs/untrained_yolo_s_acc/results.csv"
    }
}


# ---------------------------------------------------------------
# Small helpers (paths, IO, etc.)
# ---------------------------------------------------------------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def ensure_dir_with_msg(p: str, msg: str):
    if not os.path.exists(p):
        raise FileNotFoundError(msg)


def safe_copy(src: str, dst: str):
    # blunt copy helper; if same, just skip
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def list_images(folder: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def read_image_size(path: str) -> Tuple[int, int]:
    # Returns (width, height)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h

def normalize_path(path: str) -> str:
    """Ensure YOLO always sees forward slashes."""
    return os.path.abspath(path).replace("\\", "/")


def xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h) -> Tuple[float, float, float, float]:
    # YOLO format = (x_center, y_center, width, height) normalized 0 to 1
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clip values just in case
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return x_center, y_center, w, h


def iou_xyxy(a, b) -> float:
    # a,b = (xmin,ymin,xmax,ymax)
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    areaB = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = areaA + areaB - inter
    if denom <= 0:
        return 0.0
    return inter / denom


# ---------------------------------------------------------------
# Dataset prep (convert CSV to YOLOv8 structure)
# ---------------------------------------------------------------
def load_and_group_training_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Force numeric (sometimes CSVs store as strings)
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])

    # Group all boxes per image
    grouped = df.groupby("image").agg(list).reset_index()
    return grouped


def build_yolo_dataset_structure(grouped_df: pd.DataFrame,
                                 train_images_root: str,
                                 test_images_root: str,
                                 workspace_root: str,
                                 val_ratio: float = 0.1,
                                 seed: int = 42) -> Dict[str, str]:
    """
    Creates:
      workspace/
        dataset.yaml
        train/images/*.jpg
        train/labels/*.txt
        val/images/*.jpg
        val/labels/*.txt
        test/images/*.jpg
    Returns paths via dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset_dir = os.path.join(workspace_root)
    train_images_dir = os.path.join(dataset_dir, "train", "images")
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_images_dir = os.path.join(dataset_dir, "val", "images")
    val_labels_dir = os.path.join(dataset_dir, "val", "labels")
    test_images_dir = os.path.join(dataset_dir, "test", "images")

    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir]:
        ensure_dir(d)

    # Split images into train/val
    all_image_paths = list_images(train_images_root)
    all_image_names = [os.path.basename(p) for p in all_image_paths]
    train_names, val_names = train_test_split(all_image_names, test_size=val_ratio, random_state=seed, shuffle=True)

    # Build dict for quick lookup of boxes
    # grouped_df has rows: image, xmin(list), ymin(list), xmax(list), ymax(list)
    box_map = {}
    for _, row in grouped_df.iterrows():
        img = row["image"]
        xmins = row["xmin"]; ymins = row["ymin"]; xmaxs = row["xmax"]; ymaxs = row["ymax"]
        if isinstance(xmins, list) and isinstance(ymins, list):
            boxes = list(zip(xmins, ymins, xmaxs, ymaxs))
        else:
            boxes = [(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))]
        box_map[img] = boxes

    # Helper to write YOLO .txt labels
    def write_yolo_label(img_src_path: str, img_name: str, out_label_path: str):
        img_w, img_h = read_image_size(img_src_path)
        yolo_lines = []
        for (xmin, ymin, xmax, ymax) in box_map.get(img_name, []):
            x_c, y_c, w, h = xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h)
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    # Copy train/val images + labels
    for split, names, out_img_dir, out_lbl_dir in [
        ("train", train_names, train_images_dir, train_labels_dir),
        ("val", val_names, val_images_dir, val_labels_dir)
    ]:
        print(f"Preparing {split} set with {len(names)} images ...")
        for im in names:
            src = os.path.join(train_images_root, im)
            if not os.path.exists(src):
                continue
            dst = os.path.join(out_img_dir, im)
            safe_copy(src, dst)
            lbl_dst = os.path.splitext(os.path.join(out_lbl_dir, im))[0] + ".txt"
            write_yolo_label(src, im, lbl_dst)

    # Copy test images
    test_imgs = list_images(test_images_root)
    for t in test_imgs:
        dst = os.path.join(test_images_dir, os.path.basename(t))
        safe_copy(t, dst)

    # Write dataset.yaml
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")

    # Precompute safe paths (convert Windows backslashes to forward slashes)
    dataset_path = normalize_path(dataset_dir)
    train_path = normalize_path(train_images_dir)
    val_path = normalize_path(val_images_dir)
    test_path = normalize_path(test_images_dir)

    yaml_text = f"""# Auto-generated dataset file
path: {dataset_path}
train: {train_path}
val: {val_path}
test: {test_path}

names: {json.dumps(CLASS_NAMES)}
nc: {NC}
"""
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return {
        "dataset_dir": dataset_dir,
        "dataset_yaml": dataset_yaml,
        "train_images": train_images_dir,
        "val_images": val_images_dir,
        "test_images": test_images_dir
    }


# ---------------------------------------------------------------
# Model handling (pretrained weights, training, results parsing)
# ---------------------------------------------------------------
def get_or_download_pretrained(pretrained_name: str, local_dir: str) -> str:
    """
    If ./models/<pretrained_name> exists, use it.
    Else, let Ultralytics download it (YOLO(pretrained_name)) and then attempt to copy
    the underlying weights file into ./models/ for future re-use.
    """
    ensure_dir(local_dir)
    local_path = os.path.join(local_dir, pretrained_name)
    if os.path.exists(local_path):
        print(f"Using local pretrained weights: {local_path}")
        return local_path

    print(f"Downloading pretrained weights via Ultralytics for: {pretrained_name}")
    # This will download to Ultralytics cache
    y = YOLO(pretrained_name)
    # Try to copy the resolved weight file into our local models dir
    # Newer ultralytics exposes y.ckpt_path or y.model.yaml attributes; we guard with try/except
    copied = False
    try:
        # Try best-known attributes to find source path
        src_candidates = []
        if hasattr(y, "ckpt_path") and y.ckpt_path:
            src_candidates.append(y.ckpt_path)
        if hasattr(y, "model") and hasattr(y.model, "pt_path") and y.model.pt_path:
            src_candidates.append(y.model.pt_path)
        # Try to infer from name inside the cache
        for c in src_candidates:
            if c and os.path.exists(c):
                safe_copy(c, local_path)
                copied = True
                break
    except Exception:
        pass

    if not copied:
        # Fallback: we won't fail; we'll just rely on cache path when we instantiate again
        print("Could not copy cached pretrained weights to ./models/. Will load from name directly.")

    # Return local path if copied, else return the name (which YOLO can resolve from cache)
    return local_path if os.path.exists(local_path) else pretrained_name


def train_one_model(model_name: str, pretrained_path_or_name: str, dataset_yaml_path: str,
                    run_name: str, epochs: int, batch: int, imgsz: int) -> Dict[str, str]:
    """
    Trains one YOLOv8 model and returns paths to key artifacts:
    - results_csv
    - best_weights
    - val_predictions_dir (for plots)
    """
    print(f"\n=== Training {model_name} ===")
    model = YOLO(pretrained_path_or_name)  # load pretrained
    # We fine-tune for detection (nc=1)
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=os.path.join(WORKSPACE, "runs"),
        name=run_name,
        verbose=True,
        exist_ok=True
    )

    # Ultralytics stores results under runs/detect/<run_name>/
    run_dir = os.path.join(WORKSPACE, "runs", "detect", run_name)
    results_csv = os.path.join(run_dir, "results.csv")
    best_weights = os.path.join(run_dir, "weights", "best.pt")

    # Make a friendly copy into ./models/
    ensure_dir(MODELS_DIR)
    dst_best = os.path.join(MODELS_DIR, f"{model_name}_best.pt")
    if os.path.exists(best_weights):
        safe_copy(best_weights, dst_best)
        print(f"Saved best weights for {model_name} to: {dst_best}")

    return {
        "run_dir": run_dir,
        "results_csv": results_csv,
        "best_weights": dst_best if os.path.exists(dst_best) else best_weights
    }


def read_results_csv(results_csv: str) -> pd.DataFrame:
    if not os.path.exists(results_csv):
        print(f"Warning: results.csv not found at {results_csv}. Curves will be limited.")
        return pd.DataFrame()
    df = pd.read_csv(results_csv)
    return df


# ---------------------------------------------------------------
# Inference + outputs + speed measurement
# ---------------------------------------------------------------
def predict_test_and_save_csv(weights_path: str, test_images_dir: str, out_csv: str) -> pd.DataFrame:
    """
    Runs detection on test images and writes a CSV with:
      image,xmin,ymin,xmax,ymax
    One row per predicted box (confidence threshold defaults to YOLO's default).
    """
    # Normalize weights path for YOLO
    weights_path = normalize_path(weights_path)
    test_images_dir = normalize_path(test_images_dir)

    model = YOLO(weights_path)
    test_images = list_images(test_images_dir)
    rows = []
    print(f"Running inference on {len(test_images)} test images ...")
    # Batch predict for speed
    results = model.predict(source=test_images_dir, imgsz=IMGSZ, stream=True, verbose=False)
    for r in results:
        # r.path is image path; r.boxes.xyxy is Nx4 tensor; r.boxes.conf has confidences
        im_name = os.path.basename(getattr(r, "path", ""))
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        for b in xyxy:
            xmin, ymin, xmax, ymax = [float(x) for x in b]
            rows.append([im_name, xmin, ymin, xmax, ymax])
    df = pd.DataFrame(rows, columns=["image", "xmin", "ymin", "xmax", "ymax"])
    df.to_csv(out_csv, index=False)
    print(f"Wrote test predictions: {out_csv} ({len(df)} rows)")
    return df


def sample_detection_grid(weights_path: str, images_dir: str, n: int = 8, seed: int = 123):
    """
    Returns a matplotlib Figure showing a grid of detections (with boxes).
    """
    random.seed(seed)
    imgs = list_images(images_dir)
    if len(imgs) == 0:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No images found for samples", ha='center', va='center')
        plt.axis('off')
        return fig

    picks = random.sample(imgs, min(n, len(imgs)))
    model = YOLO(weights_path)
    # Run predict individually so we can draw per image
    detections = []
    for p in picks:
        res = model.predict(source=p, imgsz=IMGSZ, verbose=False)[0]
        detections.append((p, res))

    cols = 4
    rows = math.ceil(len(detections)/cols)
    fig = plt.figure(figsize=(18, 4*rows))
    for idx, (p, r) in enumerate(detections):
        ax = fig.add_subplot(rows, cols, idx+1)
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(os.path.basename(p))
        ax.axis('off')
        # draw boxes
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (xmin, ymin, xmax, ymax) in xyxy:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                           fill=False, linewidth=2, edgecolor="red"))
    plt.tight_layout()
    return fig


def _parse_flops_from_info_string(info_str: str) -> float:
    """Try to parse GFLOPs from a variety of possible info() string formats. Returns GFLOPs or None."""
    if not info_str:
        return None
    # Look for patterns like 'GFLOPs: 12.34', 'GFlops: 12.34', 'GFLOPS: 12.34', 'GFLOPs', 'GMac'
    m = re.search(r"([0-9]+\.?[0-9]*)\s*(G|g)FLOP|([0-9]+\.?[0-9]*)\s*(G|g)FLOP|([0-9]+\.?[0-9]*)\s*(G|g)Mac", info_str)
    if m:
        # find first float in the string
        m2 = re.search(r"([0-9]+\.?[0-9]*)", info_str)
        if m2:
            try:
                val = float(m2.group(1))
                return float(val)
            except Exception:
                return None
    # fallback: look for 'GFLOPs: X'
    m3 = re.search(r"GFLOP[s]?:\s*([0-9]+\.?[0-9]*)", info_str, flags=re.IGNORECASE)
    if m3:
        return float(m3.group(1))
    # nothing found
    return None


def estimate_flops(model: YOLO, imgsz: int = 640) -> float:
    """
    Attempt multiple strategies to estimate GC FLOPs for the model. Returns GFLOPs (float) or None.
    Strategies (in order):
      - Use Ultralytics model.info(...) when available and parse the string.
      - If thop is installed, attempt to use it as a fallback.
    """
    # 1) try ultralytics info() on model.model or model itself
    try:
        # newer ultralytics: model.model.info or model.info
        info_str = None
        if hasattr(model, "model") and hasattr(model.model, "info"):
            try:
                # some versions print info; capture stdout by calling and retrieving returned string
                info_out = model.model.info(verbose=False, imgsz=imgsz)
                if isinstance(info_out, str):
                    info_str = info_out
            except Exception:
                # try model.info
                try:
                    info_out2 = model.info(verbose=False, imgsz=imgsz)
                    if isinstance(info_out2, str):
                        info_str = info_out2
                except Exception:
                    info_str = None
        else:
            try:
                info_out2 = model.info(verbose=False, imgsz=imgsz)
                if isinstance(info_out2, str):
                    info_str = info_out2
            except Exception:
                info_str = None

        if info_str:
            flops = _parse_flops_from_info_string(info_str)
            if flops is not None:
                return flops
    except Exception:
        pass

    # 2) try thop if available
    try:
        from thop import profile
        import torch
        model_pt = model.model if hasattr(model, "model") else None
        if model_pt is None:
            return None
        model_pt.eval()
        device = next(model_pt.parameters()).device if len(list(model_pt.parameters()))>0 else torch.device('cpu')
        input_tensor = torch.randn(1, 3, imgsz, imgsz).to(device)
        macs, params = profile(model_pt, inputs=(input_tensor,), verbose=False)
        # thop reports MACs; FLOPs often considered 2*MACs but convention varies; we'll convert MACs to FLOPs by *2
        flops = float(macs) * 2.0 / 1e9
        return flops
    except Exception:
        pass

    return None


def measure_model_speed_and_flops(weights_path: str, images_dir: str, imgsz: int = IMGSZ,
                                  warmup: int = 10, max_images: int = None) -> Dict:
    """
    Measures per-image inference times (ms) for every image in images_dir (or up to max_images if set).
    Also attempts to estimate FLOPs (GFLOPs) for the model.

    Returns dictionary with keys: latencies_ms (list), p50, p95, median_ms, total_time_s, fps_median, throughput_fps, flops_g
    """
    weights_path = normalize_path(weights_path)
    images_dir = normalize_path(images_dir)
    imgs = list_images(images_dir)
    if max_images is not None and max_images > 0:
        imgs = imgs[:max_images]

    model = YOLO(weights_path)

    # warm-up using the first available image (or a blank tensor if none)
    if len(imgs) > 0:
        warm_img = imgs[0]
        print(f"Warming up {weights_path} with {warmup} runs on {os.path.basename(warm_img)} ...")
        for _ in range(warmup):
            try:
                _ = model.predict(source=warm_img, imgsz=imgsz, verbose=False)[0]
            except Exception:
                pass
    else:
        print("No images to measure speed on for", images_dir)

    latencies = []
    total_start = time.perf_counter()
    for p in imgs:
        try:
            t0 = time.perf_counter()
            _ = model.predict(source=p, imgsz=imgsz, verbose=False)[0]
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)
        except Exception as e:
            print(f"Warning: prediction failed for {p}: {e}")
    total_end = time.perf_counter()

    metrics = {}
    if len(latencies) > 0:
        metrics['latencies_ms'] = latencies
        metrics['p50_ms'] = float(np.percentile(latencies, 50))
        metrics['p95_ms'] = float(np.percentile(latencies, 95))
        metrics['median_ms'] = float(np.median(latencies))
        metrics['mean_ms'] = float(np.mean(latencies))
        total_time = float(total_end - total_start)
        metrics['total_time_s'] = total_time
        metrics['throughput_fps'] = float(len(latencies) / total_time) if total_time > 0 else None
        metrics['fps_from_median'] = float(1000.0 / metrics['median_ms']) if metrics['median_ms'] > 0 else None
    else:
        metrics['latencies_ms'] = []
        metrics['p50_ms'] = None
        metrics['p95_ms'] = None
        metrics['median_ms'] = None
        metrics['mean_ms'] = None
        metrics['total_time_s'] = 0.0
        metrics['throughput_fps'] = None
        metrics['fps_from_median'] = None

    # FLOPs
    try:
        flops = estimate_flops(model, imgsz=imgsz)
        metrics['flops_g'] = float(flops) if flops is not None else None
    except Exception:
        metrics['flops_g'] = None

    return metrics


# ---------------------------------------------------------------
# Report generation (PDF with per-model sections + speed graphs)
# ---------------------------------------------------------------
def plot_training_curves(results_df: pd.DataFrame, model_label: str):
    """
    Creates a figure with training/validation curves.
    We try to show:
      - train/val box_loss if available
      - metrics/precision, metrics/recall, metrics/mAP50 if available
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.set_title(f"{model_label} — Loss over Epochs")
    if not results_df.empty:
        # Ultralytics 'results.csv' often has columns like:
        #  'epoch','train/box_loss','train/cls_loss','val/box_loss','val/cls_loss',
        #  'metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)', ...
        epoch = results_df.get("epoch", pd.Series(range(len(results_df))))
        if "train/box_loss" in results_df.columns:
            ax1.plot(epoch, results_df["train/box_loss"], label="Train Box Loss")
        if "val/box_loss" in results_df.columns:
            ax1.plot(epoch, results_df["val/box_loss"], label="Val Box Loss")
        if "train/cls_loss" in results_df.columns:
            ax1.plot(epoch, results_df["train/cls_loss"], label="Train Cls Loss")
        if "val/cls_loss" in results_df.columns:
            ax1.plot(epoch, results_df["val/cls_loss"], label="Val Cls Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2.set_title(f"{model_label} — Metrics over Epochs")
    if not results_df.empty:
        epoch = results_df.get("epoch", pd.Series(range(len(results_df))))
        for col, nice in [
            ("metrics/precision(B)", "Precision(B)"),
            ("metrics/recall(B)", "Recall(B)"),
            ("metrics/mAP50(B)", "mAP@50(B)"),
            ("metrics/mAP50-95(B)", "mAP@50-95(B)")
        ]:
            if col in results_df.columns:
                ax2.plot(epoch, results_df[col], label=nice)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    return fig


def generate_pdf_report(per_model_artifacts: Dict[str, Dict[str, str]], out_pdf: str,
                        dataset_preview_dir: str):
    """
    For each model:
      - add training curves page
      - add grid of detection samples page
    Also include a title/summary page and additional pages with speed graphs for
    validation and test metrics (inference ms, FPS, FLOPs).
    """
    # Aggregate metrics for plotting
    models = list(per_model_artifacts.keys())

    # Build metric tables for val and test
    val_meds = []
    val_p95 = []
    val_fps = []
    val_flops = []

    test_meds = []
    test_p95 = []
    test_fps = []
    test_flops = []

    for m in models:
        art = per_model_artifacts[m]
        metrics = art.get('metrics', {})
        val = metrics.get('val', {}) if metrics else {}
        test = metrics.get('test', {}) if metrics else {}
        val_meds.append(val.get('median_ms'))
        val_p95.append(val.get('p95_ms'))
        val_fps.append(val.get('fps_from_median'))
        val_flops.append(val.get('flops_g'))

        test_meds.append(test.get('median_ms'))
        test_p95.append(test.get('p95_ms'))
        test_fps.append(test.get('fps_from_median'))
        test_flops.append(test.get('flops_g'))

    with PdfPages(out_pdf) as pdf:
        # Title page
        fig = plt.figure(figsize=(11.7, 8.3))
        plt.text(0.5, 0.8, "Car Object Detection Report", ha='center', va='center', fontsize=22, weight='bold')
        lines = [
            "Models: YOLO (yolov8s) and FastYOLO (yolov8n) and UnpretrainedYOLO",
            "Dataset: Kaggle Car Object Detection (converted to YOLO format)",
            "Outputs: test_predictions.csv, trained weights, curves, sample detections, speed metrics",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        for i, line in enumerate(lines):
            plt.text(0.5, 0.6 - i*0.06, line, ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig); plt.close(fig)

        # Per-model sections (curves + samples)
        for model_label, art in per_model_artifacts.items():
            # Curves
            df = read_results_csv(art.get("results_csv", ""))
            fig_curves = plot_training_curves(df, model_label)
            pdf.savefig(fig_curves)
            plt.close(fig_curves)

            # Sample detections (use VAL images to visualize, fallback to test)
            vis_dir = dataset_preview_dir
            fig_samples = sample_detection_grid(art.get("best_weights"), vis_dir, n=24)
            fig_samples.suptitle(f"{model_label} — Sample Detections", fontsize=14, y=1.02)
            pdf.savefig(fig_samples)
            plt.close(fig_samples)

        # ===== Additional pages: Speed graphs =====
        # Validation metrics page
        fig_val = plt.figure(figsize=(11.7, 8.3))
        fig_val.suptitle("Speed Metrics — Validation Set", fontsize=16)
        ax1 = fig_val.add_subplot(3, 1, 1)
        ax2 = fig_val.add_subplot(3, 1, 2)
        ax3 = fig_val.add_subplot(3, 1, 3)

        x = np.arange(len(models))
        # median latency (ms)
        ax1.bar(x, [v if v is not None else 0 for v in val_meds])
        ax1.set_xticks(x); ax1.set_xticklabels(models)
        ax1.set_ylabel('Median Latency (ms)')
        # fps
        ax2.bar(x, [v if v is not None else 0 for v in val_fps])
        ax2.set_xticks(x); ax2.set_xticklabels(models)
        ax2.set_ylabel('FPS (from median latency)')
        # flops
        ax3.bar(x, [v if v is not None else 0 for v in val_flops])
        ax3.set_xticks(x); ax3.set_xticklabels(models)
        ax3.set_ylabel('FLOPs (G)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig_val); plt.close(fig_val)

        # Test metrics page
        fig_test = plt.figure(figsize=(11.7, 8.3))
        fig_test.suptitle("Speed Metrics — Test Set", fontsize=16)
        ax1 = fig_test.add_subplot(3, 1, 1)
        ax2 = fig_test.add_subplot(3, 1, 2)
        ax3 = fig_test.add_subplot(3, 1, 3)

        x = np.arange(len(models))
        ax1.bar(x, [v if v is not None else 0 for v in test_meds])
        ax1.set_xticks(x); ax1.set_xticklabels(models)
        ax1.set_ylabel('Median Latency (ms)')

        ax2.bar(x, [v if v is not None else 0 for v in test_fps])
        ax2.set_xticks(x); ax2.set_xticklabels(models)
        ax2.set_ylabel('FPS (from median latency)')

        ax3.bar(x, [v if v is not None else 0 for v in test_flops])
        ax3.set_xticks(x); ax3.set_xticklabels(models)
        ax3.set_ylabel('FLOPs (G)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig_test); plt.close(fig_test)

    print(f"Saved report: {out_pdf}")


def generate_personal_images_report(per_model_artifacts: Dict[str, Dict[str, str]],
                                    out_pdf: str = "personal_images_report.pdf",
                                    personal_images_dir: str = PERSONAL_DIR,
                                    imgsz: int = IMGSZ,
                                    cols: int = 3,
                                    rows: int = 3) -> str:
    """
    Create a PDF (out_pdf) containing detection results on ALL images in personal_images_dir
    for each model in per_model_artifacts. Each model's outputs are clearly separated by a
    model-title page (only the model title) followed by grid pages of images with boxes.
    IMPORTANT: images will NOT have titles or filenames shown — only the model/title pages
    separate model outputs.

    Returns the path to the written PDF (out_pdf).
    """
    # verify input dir
    if not os.path.exists(personal_images_dir):
        raise FileNotFoundError(f"Personal images directory not found: {personal_images_dir}")

    imgs = list_images(personal_images_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in PERSONAL_DIR: {personal_images_dir}")

    per_page = cols * rows

    with PdfPages(out_pdf) as pdf:
        # Iterate models in order of keys in per_model_artifacts
        for model_label, art in per_model_artifacts.items():
            weights = art.get("best_weights")
            if weights is None or not os.path.exists(weights):
                # Try to fall back to the weights path anyway (YOLO can accept cache name)
                print(f"Warning: weights for {model_label} not found at {weights}. Attempting to load anyway.")
            # Add a single title page for this model (only the model title)
            fig_title = plt.figure(figsize=(11.7, 8.3))
            plt.text(0.5, 0.5, model_label, ha='center', va='center', fontsize=28, weight='bold')
            plt.axis('off')
            pdf.savefig(fig_title);
            plt.close(fig_title)

            # Load model once per model
            ymodel = YOLO(normalize_path(weights) if weights else weights)

            # Process images in consistent order
            for i in range(0, len(imgs), per_page):
                chunk = imgs[i:i + per_page]
                rows_actual = math.ceil(len(chunk) / cols)
                fig = plt.figure(figsize=(cols * 5, rows_actual * 4))
                for idx, img_path in enumerate(chunk):
                    ax = fig.add_subplot(rows_actual, cols, idx + 1)
                    # read and show image
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        # show blank if unreadable
                        ax.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                        ax.axis('off')
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.axis('off')  # NO image titles as requested

                    # run predict for this image and draw boxes
                    try:
                        res = ymodel.predict(source=img_path, imgsz=imgsz, verbose=False)[0]
                    except Exception as e:
                        # if prediction fails, still continue
                        print(f"Prediction failed for {img_path} with model {model_label}: {e}")
                        continue

                    if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        for (xmin, ymin, xmax, ymax) in xyxy:
                            # draw rectangle in display coordinates
                            ax.add_patch(plt.Rectangle((xmin, ymin),
                                                       xmax - xmin,
                                                       ymax - ymin,
                                                       fill=False, linewidth=2, edgecolor="red"))
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved personal images report: {out_pdf}")
    return out_pdf


def train():
    # Sanity checks - must exist for program to work
    ensure_dir_with_msg(TRAIN_IMAGES_DIR, f"Expected training images at {TRAIN_IMAGES_DIR}")
    ensure_dir_with_msg(TEST_IMAGES_DIR, f"Expected testing images at {TEST_IMAGES_DIR}")
    ensure_dir_with_msg(TRAIN_CSV, f"Expected CSV at {TRAIN_CSV}")

    ensure_dir(WORKSPACE)
    ensure_dir(MODELS_DIR)

    # 1) Load CSV and build YOLO dataset structure
    print("Loading training CSV and preparing YOLO dataset ...")
    grouped = load_and_group_training_csv(TRAIN_CSV)
    ds_paths_local = build_yolo_dataset_structure(
        grouped_df=grouped,
        train_images_root=TRAIN_IMAGES_DIR,
        test_images_root=TEST_IMAGES_DIR,
        workspace_root=WORKSPACE,
        val_ratio=VAL_SPLIT,
        seed=RANDOM_SEED
    )
    dataset_yaml = ds_paths_local["dataset_yaml"]

    # 2) Train models
    trained = {}
    for cfg in MODEL_CONFIGS:
        # Ensure pretrained weights locally (or use cache)
        pretrained_local_or_name = get_or_download_pretrained(cfg["pretrained"], MODELS_DIR)
        art = train_one_model(
            model_name=cfg["name"],
            pretrained_path_or_name=pretrained_local_or_name,
            dataset_yaml_path=dataset_yaml,
            run_name=cfg["run_name"],
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMGSZ
        )
        trained[cfg["name"]] = art

    # Update global per_model mapping if needed
    for k, v in trained.items():
        per_model[k] = v


def test():
    # 3) Run inference on TEST and write CSV (one per model, so you can compare)
    test_csvs = {}
    # Ensure ds_paths exist on disk
    ensure_dir_with_msg(ds_paths['test_images'], f"Expected dataset test images at {ds_paths['test_images']}")
    # Use VAL images for visualization if available
    val_dir = ds_paths.get('val_images')
    if not os.path.exists(val_dir) or len(list_images(val_dir)) == 0:
        # fallback
        val_dir = ds_paths.get('test_images')

    for cfg in MODEL_CONFIGS:
        model_label = cfg["name"]
        weights = per_model[model_label]["best_weights"]
        out_csv = f"test_predictions_{model_label}.csv"
        predict_test_and_save_csv(weights, ds_paths["test_images"], out_csv)
        test_csvs[model_label] = out_csv

    # 3.5) Measure speed + flops for VAL and TEST (per-model)
    print("\nMeasuring speed and FLOPs for each model on validation and test sets ...")
    for cfg in MODEL_CONFIGS:
        model_label = cfg["name"]
        weights = per_model[model_label]["best_weights"]
        # val
        val_metrics = measure_model_speed_and_flops(weights, ds_paths['val_images'], imgsz=IMGSZ, warmup=10)
        # test
        test_metrics = measure_model_speed_and_flops(weights, ds_paths['test_images'], imgsz=IMGSZ, warmup=10)
        # attach
        per_model[model_label]['metrics'] = {'val': val_metrics, 'test': test_metrics}

    # dump metrics to JSON for easy inspection
    try:
        with open(SPEED_JSON, 'w', encoding='utf-8') as f:
            json.dump({k: v.get('metrics', {}) for k, v in per_model.items()}, f, indent=2)
        print(f"Wrote speed metrics to {SPEED_JSON}")
    except Exception as e:
        print(f"Could not write speed metrics JSON: {e}")

    # 4) Generate PDF report with curves + samples (use VAL images for samples)
    dataset_preview = val_dir
    generate_pdf_report(per_model, REPORT_PDF, dataset_preview_dir=dataset_preview)
    personal_report = generate_personal_images_report(
        per_model,
        out_pdf="personal_images_report.pdf",
        personal_images_dir=PERSONAL_DIR
    )

    print("\nAll done!")
    print("Artifacts created:")
    for m, art in per_model.items():
        print(f"  - {m} best weights: {art['best_weights']}")
        print(f"  - {m} results.csv:  {art['results_csv']}")
        print(f"  - Test CSV:         {test_csvs[m]}")
        if 'metrics' in art:
            print(f"    - Metrics (val): median={art['metrics']['val'].get('median_ms')} ms, p95={art['metrics']['val'].get('p95_ms')} ms, flops={art['metrics']['val'].get('flops_g')}")
            print(f"    - Metrics (test): median={art['metrics']['test'].get('median_ms')} ms, p95={art['metrics']['test'].get('p95_ms')} ms, flops={art['metrics']['test'].get('flops_g')}")
    print(f"  - Personal images report written:       {personal_report}")
    print(f"  - Report PDF:       {REPORT_PDF}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    # Ensuring moderation gets the same results as our run
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # train()
    test()


if __name__ == "__main__":
    main()
