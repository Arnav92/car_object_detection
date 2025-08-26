#!/usr/bin/env python3
"""
Full Car Detection pipeline with per-epoch speed profiling.

- Rectangular training (rect=True) and save_period=1 to ensure epoch checkpoints exist.
- After training, scan the run/weights folder for epoch checkpoints (epoch1.pt, epoch2.pt, ...).
- For each epoch checkpoint, run inference on the full VAL set and record median latency and FPS.
- Augment results.csv with speed columns and save a speed_per_epoch_<run_name>.csv.
- Produce a PDF report with loss/metrics and, below them, Speed vs Epochs (median ms and FPS).
- Keep final bar charts for validation/test and write speed_metrics.json for final summaries.

NOTE: Running per-epoch speed on the full val set is expensive. Use a subset if needed.
"""
import os
import math
import glob
import time
import json
import shutil
import random
import warnings
import re
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

# plotting style
font = {'weight': 'bold', 'size': 11}
matplotlib.rc('font', **font)

# -------------------------------
# User variables (adjust as needed)
# -------------------------------
DATA_DIR = os.path.join("data")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "testing_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train_solution_bounding_boxes (1).csv")
PERSONAL_DIR = os.path.join(DATA_DIR, "personal_images")

WORKSPACE = "workspace"
MODELS_DIR = "models"
REPORT_PDF = "car_detection_report.pdf"
SPEED_JSON = "speed_metrics.json"

# Training settings
EPOCHS = 20
BATCH_SIZE = 16
IMGSZ = 640  # still used for some ops; rectangular training preserves shape
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Models to train/evaluate
MODEL_CONFIGS = [
    {"name": "YOLO", "pretrained": "yolov8s.pt", "run_name": "yolo_s_acc"},
    {"name": "FastYOLO", "pretrained": "yolov8n.pt", "run_name": "yolo_n_fast"},
    {"name": "UnpretrainedYOLO", "pretrained": "yolov8s.yaml", "run_name": "untrained_yolo_s_acc"}
]

CLASS_NAMES = ["car"]
NC = len(CLASS_NAMES)

base_dir = "workspace/"
ds_paths = {
    "train_images": f"{base_dir}/train/images",
    "val_images": f"{base_dir}/val/images",
    "test_images": f"{base_dir}/test/images"
}

# defaults (will be updated after training)
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

# -------------------------------
# Helpers
# -------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def ensure_dir_with_msg(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(msg)

def safe_copy(src: str, dst: str):
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
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h

def normalize_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")

# convert xyxy to YOLO normalized coordinates
def xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return x_center, y_center, w, h

# -------------------------------
# Dataset prep
# -------------------------------
def load_and_group_training_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    grouped = df.groupby("image").agg(list).reset_index()
    return grouped

def build_yolo_dataset_structure(grouped_df: pd.DataFrame,
                                 train_images_root: str,
                                 test_images_root: str,
                                 workspace_root: str,
                                 val_ratio: float = 0.1,
                                 seed: int = 42) -> Dict[str, str]:
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

    all_image_paths = list_images(train_images_root)
    all_image_names = [os.path.basename(p) for p in all_image_paths]
    train_names, val_names = train_test_split(all_image_names, test_size=val_ratio, random_state=seed, shuffle=True)

    # prepare box map
    box_map = {}
    for _, row in grouped_df.iterrows():
        img = row["image"]
        xmins = row["xmin"]; ymins = row["ymin"]; xmaxs = row["xmax"]; ymaxs = row["ymax"]
        if isinstance(xmins, list) and isinstance(ymins, list):
            boxes = list(zip(xmins, ymins, xmaxs, ymaxs))
        else:
            boxes = [(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))]
        box_map[img] = boxes

    def write_yolo_label(img_src_path: str, img_name: str, out_label_path: str):
        img_w, img_h = read_image_size(img_src_path)
        yolo_lines = []
        for (xmin, ymin, xmax, ymax) in box_map.get(img_name, []):
            x_c, y_c, w, h = xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h)
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

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

    # copy test images
    test_imgs = list_images(test_images_root)
    for t in test_imgs:
        dst = os.path.join(test_images_dir, os.path.basename(t))
        safe_copy(t, dst)

    # write dataset.yaml
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
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

# -------------------------------
# Model handling (download / train)
# -------------------------------
def get_or_download_pretrained(pretrained_name: str, local_dir: str) -> str:
    ensure_dir(local_dir)
    local_path = os.path.join(local_dir, pretrained_name)
    if os.path.exists(local_path):
        print(f"Using local pretrained weights: {local_path}")
        return local_path

    print(f"Downloading pretrained weights via Ultralytics for: {pretrained_name}")
    y = YOLO(pretrained_name)
    copied = False
    try:
        src_candidates = []
        if hasattr(y, "ckpt_path") and y.ckpt_path:
            src_candidates.append(y.ckpt_path)
        if hasattr(y, "model") and hasattr(y.model, "pt_path") and y.model.pt_path:
            src_candidates.append(y.model.pt_path)
        for c in src_candidates:
            if c and os.path.exists(c):
                safe_copy(c, local_path)
                copied = True
                break
    except Exception:
        pass

    if not copied:
        print("Could not copy cached pretrained weights to ./models/. Will load from name directly.")

    return local_path if os.path.exists(local_path) else pretrained_name

# helper to infer epoch number from filename
def _epoch_number_from_filename(fn: str):
    name = os.path.basename(fn)
    # skip best/last
    if name.lower().startswith("best") or name.lower().startswith("last"):
        return None
    # patterns: epoch10.pt, epoch_10.pt, epoch-10.pt
    m = re.search(r"epoch[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # fallback: look for a standalone number before .pt
    m2 = re.search(r"(^|\D)(\d+)\.pt$", name)
    if m2:
        return int(m2.group(2))
    return None

def train_one_model(model_name: str, pretrained_path_or_name: str, dataset_yaml_path: str,
                    run_name: str, epochs: int, batch: int, imgsz: int,
                    save_period: int = 1, measure_speed_after_each_epoch: bool = True,
                    val_images_dir: str = None, warmup: int = 4) -> Dict[str, str]:
    """
    Trains one YOLOv8 model with rectangular training and save_period to capture epoch ckpts.
    Optionally measures speed after training by iterating over weights/epoch*.pt.
    """
    print(f"\n=== Training {model_name} ===")
    model = YOLO(pretrained_path_or_name)

    # Train: rect=True preserves rectangular images; save_period ensures epoch checkpoints
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        rect=True,
        save_period=save_period,
        project=os.path.join(WORKSPACE, "runs"),
        name=run_name,
        verbose=True,
        exist_ok=True
    )

    run_dir = os.path.join(WORKSPACE, "runs", "detect", run_name)
    results_csv = os.path.join(run_dir, "results.csv")
    weights_dir = os.path.join(run_dir, "weights")
    best_weights = os.path.join(weights_dir, "best.pt")

    ensure_dir(MODELS_DIR)
    dst_best = os.path.join(MODELS_DIR, f"{model_name}_best.pt")
    if os.path.exists(best_weights):
        safe_copy(best_weights, dst_best)
        print(f"Saved best weights for {model_name} to: {dst_best}")

    art = {
        "run_dir": run_dir,
        "results_csv": results_csv,
        "best_weights": dst_best if os.path.exists(dst_best) else best_weights
    }

    # If requested, and if there are epoch files, measure speed per epoch (val)
    if measure_speed_after_each_epoch and val_images_dir:
        if not os.path.exists(weights_dir):
            print(f"Warning: weights directory not found: {weights_dir}")
        else:
            # Find epoch files
            epoch_map = {}
            for p in glob.glob(os.path.join(weights_dir, "*.pt")):
                ep = _epoch_number_from_filename(p)
                if ep is not None:
                    epoch_map[ep] = p
            if len(epoch_map) == 0:
                print(f"No epoch checkpoint files (epoch*.pt) found in {weights_dir}. Skipping per-epoch speed measurement.")
            else:
                epoch_nums = sorted(epoch_map.keys())
                print(f"Found {len(epoch_nums)} epoch checkpoint files for run {run_name}. Measuring speed for each epoch (this may take a while).")
                speed_rows = []
                for ep in epoch_nums:
                    wpath = epoch_map[ep]
                    print(f"Measuring epoch {ep} using {os.path.basename(wpath)} ...")
                    metrics = measure_model_speed_and_flops(wpath, val_images_dir, imgsz=imgsz, warmup=warmup)
                    speed_rows.append({
                        "epoch_ckpt": ep,
                        # store median for plotting and fps
                        "speed/median_ms": metrics.get("median_ms"),
                        "speed/p95_ms": metrics.get("p95_ms"),
                        "speed/mean_ms": metrics.get("mean_ms"),
                        "speed/fps_from_median": metrics.get("fps_from_median"),
                        "speed/throughput_fps": metrics.get("throughput_fps"),
                        "speed/flops_g": metrics.get("flops_g")
                    })

                # Save CSV with epoch checkpoints (epoch ckpt numbering as in filename)
                speed_df = pd.DataFrame(sorted(speed_rows, key=lambda r: r["epoch_ckpt"]))
                speed_csv = os.path.join(run_dir, f"speed_per_epoch_{run_name}.csv")
                try:
                    speed_df.to_csv(speed_csv, index=False)
                    print(f"Wrote per-epoch speed CSV: {speed_csv}")
                    art["speed_per_epoch_csv"] = speed_csv
                except Exception as e:
                    print(f"Could not write per-epoch speed CSV: {e}")

                # Augment results.csv with speed columns aligned to epoch index
                if os.path.exists(results_csv):
                    try:
                        res_df = pd.read_csv(results_csv)
                        # Ensure epoch column exists: many results.csv have 'epoch' starting at 0
                        if "epoch" not in res_df.columns:
                            res_df.insert(0, "epoch", list(range(len(res_df))))

                        # Create new speed columns if not present
                        speed_cols = ["speed/val_median_ms", "speed/val_p95_ms", "speed/val_mean_ms",
                                      "speed/val_fps_from_median", "speed/val_throughput_fps", "speed/val_flops_g"]
                        for c in speed_cols:
                            if c not in res_df.columns:
                                res_df[c] = np.nan

                        # Map epoch_ckpt -> candidate epoch row index in results.csv
                        # Heuristic: epoch_ckpt (1-based) corresponds to results.csv epoch (0-based) as ep_ckpt - 1
                        for _, row in speed_df.iterrows():
                            ep_ckpt = int(row["epoch_ckpt"])
                            candidate_epochs = [ep_ckpt - 1, ep_ckpt, ep_ckpt + 1]  # try a few offsets
                            found = False
                            for cand in candidate_epochs:
                                matches = res_df.index[res_df["epoch"] == cand].tolist()
                                if len(matches) > 0:
                                    idx0 = matches[0]
                                    res_df.at[idx0, "speed/val_median_ms"] = row["speed/median_ms"]
                                    res_df.at[idx0, "speed/val_p95_ms"] = row["speed/p95_ms"]
                                    res_df.at[idx0, "speed/val_mean_ms"] = row["speed/mean_ms"]
                                    res_df.at[idx0, "speed/val_fps_from_median"] = row["speed/fps_from_median"]
                                    res_df.at[idx0, "speed/val_throughput_fps"] = row["speed/throughput_fps"]
                                    res_df.at[idx0, "speed/val_flops_g"] = row["speed/flops_g"]
                                    found = True
                                    break
                            if not found:
                                # Append a new row with epoch = ep_ckpt - 1
                                newrow = {c: np.nan for c in res_df.columns}
                                # put epoch number in the epoch column
                                newrow["epoch"] = ep_ckpt - 1
                                newrow["speed/val_median_ms"] = row["speed/median_ms"]
                                newrow["speed/val_p95_ms"] = row["speed/p95_ms"]
                                newrow["speed/val_mean_ms"] = row["speed/mean_ms"]
                                newrow["speed/val_fps_from_median"] = row["speed/fps_from_median"]
                                newrow["speed/val_throughput_fps"] = row["speed/throughput_fps"]
                                newrow["speed/val_flops_g"] = row["speed/flops_g"]
                                res_df = pd.concat([res_df, pd.DataFrame([newrow])], ignore_index=True, sort=False)

                        # sort and save back
                        res_df = res_df.sort_values(by="epoch").reset_index(drop=True)
                        res_df.to_csv(results_csv, index=False)
                        print(f"Augmented results.csv with per-epoch speed columns: {results_csv}")
                    except Exception as e:
                        print(f"Could not augment results.csv at {results_csv}: {e}")
                else:
                    print(f"results.csv not found at {results_csv}; speed CSV created at {speed_csv}")

    return art

def read_results_csv(results_csv: str) -> pd.DataFrame:
    if not os.path.exists(results_csv):
        print(f"Warning: results.csv not found at {results_csv}. Curves will be limited.")
        return pd.DataFrame()
    return pd.read_csv(results_csv)

# -------------------------------
# Inference & FLOPs estimation
# -------------------------------
def predict_test_and_save_csv(weights_path: str, test_images_dir: str, out_csv: str) -> pd.DataFrame:
    weights_path = normalize_path(weights_path)
    test_images_dir = normalize_path(test_images_dir)
    model = YOLO(weights_path)
    test_images = list_images(test_images_dir)
    rows = []
    print(f"Running inference on {len(test_images)} test images ...")
    results = model.predict(source=test_images_dir, imgsz=IMGSZ, stream=True, verbose=False)
    for r in results:
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
    random.seed(seed)
    imgs = list_images(images_dir)
    if len(imgs) == 0:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No images found for samples", ha='center', va='center')
        plt.axis('off')
        return fig

    picks = random.sample(imgs, min(n, len(imgs)))
    model = YOLO(weights_path)
    detections = []
    for p in picks:
        res = model.predict(source=p, imgsz=IMGSZ, verbose=False)[0]
        detections.append((p, res))

    cols = 4
    rows = math.ceil(len(detections) / cols)
    fig = plt.figure(figsize=(18, 4 * rows))
    for idx, (p, r) in enumerate(detections):
        ax = fig.add_subplot(rows, cols, idx + 1)
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(os.path.basename(p))
        ax.axis('off')
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (xmin, ymin, xmax, ymax) in xyxy:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, linewidth=2, edgecolor="red"))
    plt.tight_layout()
    return fig

def _parse_flops_from_info_string(info_str: str):
    if not info_str:
        return None
    # Try several patterns
    m = re.search(r"([0-9]+\.?[0-9]*)\s*(G|g)FLOP", info_str)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.search(r"GFLOP[s]?:\s*([0-9]+\.?[0-9]*)", info_str, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    # try GMac
    m3 = re.search(r"([0-9]+\.?[0-9]*)\s*(G|g)Mac", info_str)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            pass
    # fallback: first float
    m4 = re.search(r"([0-9]+\.?[0-9]*)", info_str)
    if m4:
        try:
            return float(m4.group(1))
        except Exception:
            pass
    return None

def estimate_flops(model: YOLO, imgsz: int = 640):
    """Try ultralytics info() first; fallback to thop if available."""
    try:
        info_str = None
        if hasattr(model, "model") and hasattr(model.model, "info"):
            try:
                info_out = model.model.info(verbose=False, imgsz=imgsz)
                if isinstance(info_out, str):
                    info_str = info_out
            except Exception:
                pass
        if not info_str:
            try:
                info_out2 = model.info(verbose=False, imgsz=imgsz)
                if isinstance(info_out2, str):
                    info_str = info_out2
            except Exception:
                pass
        if info_str:
            flops = _parse_flops_from_info_string(info_str)
            if flops is not None:
                return flops
    except Exception:
        pass

    # fallback to thop if installed
    try:
        from thop import profile
        import torch
        model_pt = model.model if hasattr(model, "model") else None
        if model_pt is None:
            return None
        model_pt.eval()
        device = next(model_pt.parameters()).device if len(list(model_pt.parameters())) > 0 else torch.device('cpu')
        input_tensor = torch.randn(1, 3, imgsz, imgsz).to(device)
        macs, params = profile(model_pt, inputs=(input_tensor,), verbose=False)
        flops = float(macs) * 2.0 / 1e9
        return flops
    except Exception:
        pass

    return None

def measure_model_speed_and_flops(weights_path: str, images_dir: str, imgsz: int = IMGSZ,
                                  warmup: int = 4, max_images: int = None) -> Dict:
    """
    Measure per-image latencies (ms) across all images in images_dir (or truncated).
    Returns dict with latencies list, median, p95, mean, throughput_fps, fps_from_median, flops_g.
    """
    weights_path = normalize_path(weights_path)
    images_dir = normalize_path(images_dir)
    imgs = list_images(images_dir)
    if max_images is not None and max_images > 0:
        imgs = imgs[:max_images]

    model = YOLO(weights_path)

    if len(imgs) > 0:
        warm_img = imgs[0]
        print(f"Warm-up: running {warmup} warm runs on {os.path.basename(warm_img)} ...")
        for _ in range(warmup):
            try:
                _ = model.predict(source=warm_img, imgsz=imgsz, verbose=False)[0]
            except Exception:
                pass
    else:
        print("Warning: no images found for timing in", images_dir)

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
            print(f"Warning: predict failed for {p}: {e}")
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
        metrics['fps_from_median'] = float(1000.0 / metrics['median_ms']) if metrics['median_ms'] and metrics['median_ms'] > 0 else None
    else:
        metrics['latencies_ms'] = []
        metrics['p50_ms'] = None
        metrics['p95_ms'] = None
        metrics['median_ms'] = None
        metrics['mean_ms'] = None
        metrics['total_time_s'] = None
        metrics['throughput_fps'] = None
        metrics['fps_from_median'] = None

    # FLOPs
    try:
        flops = estimate_flops(model, imgsz=imgsz)
        metrics['flops_g'] = float(flops) if flops is not None else None
    except Exception:
        metrics['flops_g'] = None

    return metrics

# -------------------------------
# Plotting & report generation
# -------------------------------
def plot_training_curves(results_df: pd.DataFrame, model_label: str):
    """
    Creates a figure with:
      - Loss subplot (top)
      - Metrics subplot (middle)
      - Speed vs Epochs subplot (bottom): median latency (inference ms) and FPS (secondary axis)
    """
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    # Loss
    ax1.set_title(f"{model_label} — Loss over Epochs")
    if not results_df.empty:
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

    # Metrics
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

    # Speed subplot
    ax3.set_title(f"{model_label} — Speed over Epochs (median latency = inference ms)")
    epoch = results_df.get("epoch", pd.Series(range(len(results_df))))
    if "speed/val_median_ms" in results_df.columns:
        ax3.plot(epoch, results_df["speed/val_median_ms"], marker="o", label="Median Latency (inference ms)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Median Latency (ms)")
    ax3.grid(True)

    if "speed/val_fps" in results_df.columns:
        ax3b = ax3.twinx()
        ax3b.plot(epoch, results_df["speed/val_fps"], marker="x", linestyle="--", label="FPS (from median)")
        ax3b.set_ylabel("FPS")
        # combine legends
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax3.legend(loc="upper right")

    plt.tight_layout()
    return fig

def generate_pdf_report(per_model_artifacts: Dict[str, Dict[str, str]], out_pdf: str,
                        dataset_preview_dir: str):
    models = list(per_model_artifacts.keys())

    # Build final summary arrays for bar charts
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
        # Title
        fig = plt.figure(figsize=(11.7, 8.3))
        plt.text(0.5, 0.8, "Car Object Detection Report", ha='center', va='center', fontsize=22, weight='bold')
        lines = [
            "Models: YOLO (yolov8s) and FastYOLO (yolov8n) and UnpretrainedYOLO",
            "Dataset: Kaggle Car Object Detection (converted to YOLO format)",
            "Outputs: test_predictions.csv, trained weights, curves, sample detections, speed metrics",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        for i, line in enumerate(lines):
            plt.text(0.5, 0.6 - i * 0.06, line, ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig); plt.close(fig)

        # Per-model pages
        for model_label, art in per_model_artifacts.items():
            df = read_results_csv(art.get("results_csv", ""))
            fig_curves = plot_training_curves(df, model_label)
            pdf.savefig(fig_curves)
            plt.close(fig_curves)

            # samples (use val if available)
            vis_dir = dataset_preview_dir
            fig_samples = sample_detection_grid(art.get("best_weights"), vis_dir, n=24)
            fig_samples.suptitle(f"{model_label} — Sample Detections", fontsize=14, y=1.02)
            pdf.savefig(fig_samples)
            plt.close(fig_samples)

        # Summary speed bar charts (val)
        fig_val = plt.figure(figsize=(11.7, 8.3))
        fig_val.suptitle("Speed Metrics — Validation Set", fontsize=16)
        ax1 = fig_val.add_subplot(3, 1, 1)
        ax2 = fig_val.add_subplot(3, 1, 2)
        ax3 = fig_val.add_subplot(3, 1, 3)
        x = np.arange(len(models))
        ax1.bar(x, [v if v is not None else 0 for v in val_meds])
        ax1.set_xticks(x); ax1.set_xticklabels(models)
        ax1.set_ylabel('Median Latency (ms)')
        ax2.bar(x, [v if v is not None else 0 for v in val_fps])
        ax2.set_xticks(x); ax2.set_xticklabels(models)
        ax2.set_ylabel('FPS (from median)')
        ax3.bar(x, [v if v is not None else 0 for v in val_flops])
        ax3.set_xticks(x); ax3.set_xticklabels(models)
        ax3.set_ylabel('FLOPs (G)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig_val); plt.close(fig_val)

        # Summary speed bar charts (test)
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
        ax2.set_ylabel('FPS (from median)')
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
    if not os.path.exists(personal_images_dir):
        raise FileNotFoundError(f"Personal images directory not found: {personal_images_dir}")

    imgs = list_images(personal_images_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images found in PERSONAL_DIR: {personal_images_dir}")

    per_page = cols * rows
    with PdfPages(out_pdf) as pdf:
        for model_label, art in per_model_artifacts.items():
            weights = art.get("best_weights")
            fig_title = plt.figure(figsize=(11.7, 8.3))
            plt.text(0.5, 0.5, model_label, ha='center', va='center', fontsize=28, weight='bold')
            plt.axis('off')
            pdf.savefig(fig_title); plt.close(fig_title)

            ymodel = YOLO(normalize_path(weights) if weights else weights)
            for i in range(0, len(imgs), per_page):
                chunk = imgs[i:i + per_page]
                rows_actual = math.ceil(len(chunk) / cols)
                fig = plt.figure(figsize=(cols * 5, rows_actual * 4))
                for idx, img_path in enumerate(chunk):
                    ax = fig.add_subplot(rows_actual, cols, idx + 1)
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        ax.text(0.5, 0.5, "Could not read image", ha='center', va='center')
                        ax.axis('off')
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.axis('off')
                    try:
                        res = ymodel.predict(source=img_path, imgsz=imgsz, verbose=False)[0]
                    except Exception as e:
                        print(f"Prediction failed for {img_path} with model {model_label}: {e}")
                        continue
                    if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        for (xmin, ymin, xmax, ymax) in xyxy:
                            ax.add_patch(plt.Rectangle((xmin, ymin),
                                                       xmax - xmin,
                                                       ymax - ymin,
                                                       fill=False, linewidth=2, edgecolor="red"))
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Saved personal images report: {out_pdf}")
    return out_pdf

# -------------------------------
# Orchestration: train / test
# -------------------------------
def train():
    ensure_dir_with_msg(TRAIN_IMAGES_DIR, f"Expected training images at {TRAIN_IMAGES_DIR}")
    ensure_dir_with_msg(TEST_IMAGES_DIR, f"Expected testing images at {TEST_IMAGES_DIR}")
    ensure_dir_with_msg(TRAIN_CSV, f"Expected CSV at {TRAIN_CSV}")

    ensure_dir(WORKSPACE)
    ensure_dir(MODELS_DIR)

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

    trained = {}
    for cfg in MODEL_CONFIGS:
        pretrained_local_or_name = get_or_download_pretrained(cfg["pretrained"], MODELS_DIR)
        art = train_one_model(
            model_name=cfg["name"],
            pretrained_path_or_name=pretrained_local_or_name,
            dataset_yaml_path=dataset_yaml,
            run_name=cfg["run_name"],
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMGSZ,
            save_period=1,
            measure_speed_after_each_epoch=True,
            val_images_dir=ds_paths_local["val_images"],
            warmup=4
        )
        trained[cfg["name"]] = art

    for k, v in trained.items():
        per_model[k] = v

def test():
    test_csvs = {}
    ensure_dir_with_msg(ds_paths['test_images'], f"Expected dataset test images at {ds_paths['test_images']}")
    val_dir = ds_paths.get('val_images')
    if not os.path.exists(val_dir) or len(list_images(val_dir)) == 0:
        val_dir = ds_paths.get('test_images')

    # 1) Run detection on test images and save per-model CSVs
    for cfg in MODEL_CONFIGS:
        model_label = cfg["name"]
        weights = per_model[model_label]["best_weights"]
        out_csv = f"test_predictions_{model_label}.csv"
        predict_test_and_save_csv(weights, ds_paths["test_images"], out_csv)
        test_csvs[model_label] = out_csv

    # 2) Measure final val/test speed (for final/best weights) and summarize
    print("\nMeasuring speed and FLOPs for each model on validation and test sets ...")
    for cfg in MODEL_CONFIGS:
        model_label = cfg["name"]
        weights = per_model[model_label]["best_weights"]
        val_metrics = measure_model_speed_and_flops(weights, ds_paths['val_images'], imgsz=IMGSZ, warmup=6)
        test_metrics = measure_model_speed_and_flops(weights, ds_paths['test_images'], imgsz=IMGSZ, warmup=6)
        per_model[model_label]['metrics'] = {'val': val_metrics, 'test': test_metrics}

    # 3) Write consolidated speed JSON (final summaries + per-epoch val series if available)
    consolidated = {}
    for m, art in per_model.items():
        consolidated[m] = art.get('metrics', {})
        # attach per-epoch speed if present
        if art.get("speed_per_epoch_csv") and os.path.exists(art["speed_per_epoch_csv"]):
            try:
                sdf = pd.read_csv(art["speed_per_epoch_csv"])
                # Normalize to lists
                consolidated[m]["val_by_epoch"] = {
                    "epoch_ckpt": sdf["epoch_ckpt"].tolist(),
                    "median_ms": sdf["speed/median_ms"].tolist() if "speed/median_ms" in sdf.columns else [None]*len(sdf),
                    "fps": sdf["speed/fps_from_median"].tolist() if "speed/fps_from_median" in sdf.columns else [None]*len(sdf),
                    "p95_ms": sdf["speed/p95_ms"].tolist() if "speed/p95_ms" in sdf.columns else [None]*len(sdf),
                    "mean_ms": sdf["speed/mean_ms"].tolist() if "speed/mean_ms" in sdf.columns else [None]*len(sdf),
                    "flops_g": sdf["speed/flops_g"].tolist() if "speed/flops_g" in sdf.columns else [None]*len(sdf)
                }
            except Exception:
                pass

    try:
        with open(SPEED_JSON, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)
        print(f"Wrote consolidated speed metrics JSON: {SPEED_JSON}")
    except Exception as e:
        print(f"Could not write {SPEED_JSON}: {e}")

    # 4) Generate PDF report (uses val preview)
    dataset_preview = val_dir
    generate_pdf_report(per_model, REPORT_PDF, dataset_preview_dir=dataset_preview)

    # 5) Personal images report
    personal_report = generate_personal_images_report(
        per_model,
        out_pdf="personal_images_report.pdf",
        personal_images_dir=PERSONAL_DIR
    )

    print("\nAll done!")
    for m, art in per_model.items():
        print(f"  - {m} best weights: {art['best_weights']}")
        print(f"  - {m} results.csv:  {art['results_csv']}")
        print(f"  - Test CSV:         {test_csvs.get(m)}")
        if 'metrics' in art:
            print(f"    - Metrics (val): median={art['metrics']['val'].get('median_ms')} ms, p95={art['metrics']['val'].get('p95_ms')} ms, flops={art['metrics']['val'].get('flops_g')}")
            print(f"    - Metrics (test): median={art['metrics']['test'].get('median_ms')} ms, p95={art['metrics']['test'].get('p95_ms')} ms, flops={art['metrics']['test'].get('flops_g')}")
        if art.get("speed_per_epoch_csv"):
            print(f"    - Per-epoch speed CSV: {art['speed_per_epoch_csv']}")
    print(f"  - Personal images report written:       {personal_report}")
    print(f"  - Report PDF:       {REPORT_PDF}")

# -------------------------------
# Main
# -------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train() # <-- Uncomment if you want to train
    test()

if __name__ == "__main__":
    main()