# What this script does:
# 1) Reads data/train_solution_bounding_boxes (1).csv
# 2) Creates a YOLOv8-style dataset folder under ./workspace/
# 3) Trains two models:
#    - "YOLO"  -> yolov8s (more accurate, slower)
#    - "FastYOLO" -> yolov8n (faster, smaller)
# 4) Saves trained models to ./models/
# 5) Runs trained models on test images and writes test_predictions.csv
# 6) Generates a PDF report with loss curves and detection samples per model
#
# Notes:
# - In the "main" function, the "train" functon is commented out. This is because
# training has already been done. If you want to examine training, uncomment that
# line and delete all files except main.py and the "data" folder
# - KFOLD is work in progress - might not do
# - If models already exist, they will be reused; otherwise they are downloaded.
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

WORKSPACE = "workspace"  # we'll build YOLOv8 format dataset here
MODELS_DIR = "models"    # store/check pretrained + trained weights here
REPORT_PDF = "car_detection_report.pdf"

# Training settings
EPOCHS = 20
BATCH_SIZE = 16
IMGSZ = 640  # YOLO standard
VAL_SPLIT = 0.1 # 90% training and 10% validation choice
RANDOM_SEED = 42
USE_KFOLD = False
KFOLD_SPLITS = 5

# Two model configs
'''MODEL_CONFIGS = [
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
        "name": "UntrainedYOLO",       # no pretraining
        "pretrained": "yolov8s.yaml",
        "run_name": "untrained_yolo_s_acc"
    }
]'''

MODEL_CONFIGS = [
    {
        "name": "UntrainedYOLO",       # no pretraining
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
'''per_model = {
    "YOLO": {
        "best_weights": "workspace/runs/yolo_s_acc/weights/best.pt",
        "results_csv": "workspace/runs/yolo_s_acc/results.csv"
    },
    "FastYOLO": {
        "best_weights": "workspace/runs/yolo_n_fast/weights/best.pt",
        "results_csv": "workspace/runs/yolo_n_fast/results.csv"
    },
    "UntrainedYOLO": {
        "best_weights": "workspace/runs/untrained_yolo_s_acc/weights/best.pt",
        "results_csv": "workspace/runs/untrained_yolo_s_acc/results.csv"
    }
}'''

per_model = {
    "UntrainedYOLO": {
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
    print(f"Grouped: {grouped}")
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
    all_image_names = grouped_df["image"].tolist()
    # Keep only those files that actually exist in training_images_root
    existing = [im for im in all_image_names if os.path.exists(os.path.join(train_images_root, im))]
    if len(existing) < len(all_image_names):
        missing = set(all_image_names) - set(existing)
        print(f"Warning: {len(missing)} images listed in CSV were not found on disk. Proceeding with {len(existing)}.")

    train_names, val_names = train_test_split(existing, test_size=val_ratio, random_state=seed, shuffle=True)

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
# Inference + outputs
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


# ---------------------------------------------------------------
# Report generation (PDF with per-model sections)
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
    Also include a title/summary page.
    """
    with PdfPages(out_pdf) as pdf:
        # Title page
        fig = plt.figure(figsize=(11.7, 8.3))
        plt.text(0.5, 0.8, "Car Object Detection Report", ha='center', va='center', fontsize=22, weight='bold')
        lines = [
            "Models: YOLO (yolov8s) and FastYOLO (yolov8n)",
            "Dataset: Kaggle Car Object Detection (converted to YOLO format)",
            "Outputs: test_predictions.csv, trained weights, curves, sample detections",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        for i, line in enumerate(lines):
            plt.text(0.5, 0.6 - i*0.06, line, ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig); plt.close(fig)

        # Per-model sections
        for model_label, art in per_model_artifacts.items():
            # Curves
            df = read_results_csv(art["results_csv"])
            fig_curves = plot_training_curves(df, model_label)
            pdf.savefig(fig_curves)
            plt.close(fig_curves)

            # Sample detections (use VAL images to visualize)
            # If val empty, fall back to test
            vis_dir = dataset_preview_dir
            fig_samples = sample_detection_grid(art["best_weights"], vis_dir, n=24)
            fig_samples.suptitle(f"{model_label} — Sample Detections", fontsize=14, y=1.02)
            pdf.savefig(fig_samples)
            plt.close(fig_samples)

    print(f"Saved report: {out_pdf}")

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
    ds_paths = build_yolo_dataset_structure(
        grouped_df=grouped,
        train_images_root=TRAIN_IMAGES_DIR,
        test_images_root=TEST_IMAGES_DIR,
        workspace_root=WORKSPACE,
        val_ratio=VAL_SPLIT,
        seed=RANDOM_SEED
    )
    dataset_yaml = ds_paths["dataset_yaml"]

    # 2) Train models
    per_model = {}
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
        per_model[cfg["name"]] = art


def test():
    # 3) Run inference on TEST and write CSV (one per model, so you can compare)
    test_csvs = {}
    for cfg in MODEL_CONFIGS:
        model_label = cfg["name"]
        weights = per_model[model_label]["best_weights"]
        out_csv = f"test_predictions_{model_label}.csv"
        predict_test_and_save_csv(weights, ds_paths["test_images"], out_csv)
        test_csvs[model_label] = out_csv

    # 4) Generate PDF report with curves + samples (use VAL images for samples)
    #    If val is empty (tiny datasets), we can also pick from train or test.
    val_dir = ds_paths["val_images"]
    use_dir = val_dir if len(list_images(val_dir)) > 0 else ds_paths["test_images"]
    generate_pdf_report(per_model, REPORT_PDF, dataset_preview_dir=use_dir)

    print("\nAll done!")
    print("Artifacts created:")
    for m, art in per_model.items():
        print(f"  - {m} best weights: {art['best_weights']}")
        print(f"  - {m} results.csv:  {art['results_csv']}")
        print(f"  - Test CSV:         {test_csvs[m]}")
    print(f"  - Report PDF:       {REPORT_PDF}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    # Ensuring moderation gets the same results as our run
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train()
    test()


if __name__ == "__main__":
    main()
