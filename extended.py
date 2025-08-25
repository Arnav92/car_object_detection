# ===============================================================
# All-in-one training + evaluation for:
#   - YOLOv8 (Ultralytics): yolov8s, yolov8n, yolov8s.yaml (from scratch)
#   - Torchvision: Faster R-CNN, RetinaNet
#   - Hugging Face: DETR (facebook/detr-resnet-50)  <-- NO torchvision detr import
#
# What this script does:
# 1) Reads CSV: data/train_solution_bounding_boxes (1).csv
# 2) Builds a YOLOv8-style dataset under ./workspace/ (train/val/test)
# 3) Trains ALL models on that dataset (fine-tune from COCO where possible)
# 4) Saves best weights + results.csv for each model to ./workspace/runs/... and ./models/
# 5) Runs inference on test images → test_predictions_<MODEL>.csv
# 6) Creates a PDF report with training curves + sample detections
#
# Notes & design choices:
# - We keep YOLO training via the Ultralytics high-level API (simple and fast).
# - For Torchvision models, we implement a clean PyTorch training loop.
# - For HF DETR, we train via its processor + model forward (no torchvision.detr).
# - We compute a simple, standard mAP@0.5 metric for validation for all non-YOLO
#   models so results.csv resembles YOLO’s convenience file.
# - Everything is single-class ("car") as per your project.
#
# IMPORTANT: Keep NumPy < 2 to avoid binary-compatibility errors in some libs.
#   pip install "numpy<2" --force-reinstall
# ===============================================================

import os
import math
import glob
import time
import json
import shutil
import random
import warnings
from typing import List, Tuple, Dict, Iterable

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Core libs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TVF
from torchvision.ops import box_convert
from torchvision.transforms import functional as TF

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- Ultralytics YOLO
from ultralytics import YOLO

# ---------------- Hugging Face DETR
from transformers import DetrImageProcessor, DetrForObjectDetection

# ---------------- Reporting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split

# Matplotlib style
font = {'weight': 'bold', 'size': 11}
matplotlib.rc('font', **font)

# ===============================================================
# User-adjustable paths & settings
# ===============================================================
DATA_DIR = "data"  # expects data/training_images, data/testing_images, data/train_solution_bounding_boxes (1).csv
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
TEST_IMAGES_DIR  = os.path.join(DATA_DIR, "testing_images")
TRAIN_CSV        = os.path.join(DATA_DIR, "train_solution_bounding_boxes (1).csv")

WORKSPACE = "workspace"
MODELS_DIR = "models"
REPORT_PDF = "car_detection_report.pdf"

# Single-class dataset
CLASS_NAMES = ["car"]
NC = 1

# Training hyper-params
RANDOM_SEED = 42
VAL_SPLIT   = 0.1     # 10% validation
IMGSZ       = 640     # YOLO train size (others use raw image sizes)
EPOCHS_YOLO = 20
EPOCHS_TV   = 12      # Torchvision models
EPOCHS_DETR = 12      # DETR
BATCH_TV    = 4
BATCH_DETR  = 4
LR_TV       = 0.0005
LR_DETR     = 0.0001
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configs (all training-based)
MODEL_CONFIGS = [
    # {"name": "YOLO",             "type": "ultralytics", "pretrained": "yolov8s.pt",   "run_name": "yolo_s_acc"},
    # {"name": "FastYOLO",         "type": "ultralytics", "pretrained": "yolov8n.pt",   "run_name": "yolo_n_fast"},
    # {"name": "UnpretrainedYOLO", "type": "ultralytics", "pretrained": "yolov8s.yaml", "run_name": "untrained_yolo_s_acc"},
    {"name": "FasterRCNN",       "type": "torchvision", "backbone": "fasterrcnn_resnet50_fpn", "run_name": "frcnn"},
    {"name": "RetinaNet",        "type": "torchvision", "backbone": "retinanet_resnet50_fpn",  "run_name": "retinanet"},
    {"name": "DETR",             "type": "hf_detr",     "pretrained_id": "facebook/detr-resnet-50", "run_name": "detr"},
]

# Per-model artifacts (we fill as we go)
per_model = {cfg["name"]: {"best_weights": "", "results_csv": ""} for cfg in MODEL_CONFIGS}

# Derived paths for the dataset we’ll build
DS = {
    "dataset_dir": os.path.join(WORKSPACE),
    "train_images": os.path.join(WORKSPACE, "train", "images"),
    "train_labels": os.path.join(WORKSPACE, "train", "labels"),
    "val_images":   os.path.join(WORKSPACE, "val", "images"),
    "val_labels":   os.path.join(WORKSPACE, "val", "labels"),
    "test_images":  os.path.join(WORKSPACE, "test", "images"),
    "yaml":         os.path.join(WORKSPACE, "dataset.yaml"),
}

# ===============================================================
# Small filesystem helpers
# ===============================================================
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def ensure_dir_with_msg(p: str, msg: str):
    if not os.path.exists(p):
        raise FileNotFoundError(msg)

def safe_copy(src: str, dst: str):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def list_images(folder: str) -> List[str]:
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
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

# ===============================================================
# Bounding-box utils + metrics (for consistent eval across models)
# ===============================================================
def xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h):
    x_c = (xmin + xmax) / 2.0 / img_w
    y_c = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w   = min(max(w,   0.0), 1.0)
    h   = min(max(h,   0.0), 1.0)
    return x_c, y_c, w, h

def iou_xyxy(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    interW, interH = max(0.0, xB - xA), max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    areaB = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    denom = areaA + areaB - inter
    if denom <= 0:
        return 0.0
    return inter / denom

def coco_to_xyxy(b):  # [x,y,w,h] → [xmin,ymin,xmax,ymax]
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def compute_map50(all_preds, all_gts, iou_th=0.5):
    """
    Simple mAP@0.5 for single-class:
    - all_preds: list of (img_id, score, [xmin,ymin,xmax,ymax])
    - all_gts:   dict img_id -> list of gt boxes [xmin,ymin,xmax,ymax]
    """
    # Sort predictions by score (desc)
    all_preds = sorted(all_preds, key=lambda x: x[1], reverse=True)
    tp, fp = [], []
    matched = {img_id: np.zeros(len(all_gts.get(img_id, [])), dtype=bool) for img_id in all_gts.keys()}
    total_gts = sum(len(v) for v in all_gts.values())

    for img_id, score, box in all_preds:
        gts = all_gts.get(img_id, [])
        ious = [iou_xyxy(box, gt) for gt in gts]
        if len(ious) and (max(ious) >= iou_th):
            j = int(np.argmax(ious))
            if not matched[img_id][j]:
                tp.append(1.0); fp.append(0.0)
                matched[img_id][j] = True
            else:
                tp.append(0.0); fp.append(1.0)
        else:
            tp.append(0.0); fp.append(1.0)

    if len(tp) == 0:
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recalls = tp / max(total_gts, 1e-9)
    precisions = tp / np.maximum(tp + fp, 1e-9)

    # 11-point interpolation (VOC-style)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precisions[recalls >= thr].max() if np.any(recalls >= thr) else 0.0
        ap += p
    ap /= 11.0
    return float(ap)

# ===============================================================
# Dataset build from CSV → YOLOv8 structure
# ===============================================================
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
    random.seed(seed); np.random.seed(seed)
    for p in [DS["train_images"], DS["train_labels"], DS["val_images"], DS["val_labels"], DS["test_images"]]:
        ensure_dir(p)

    # train/val split
    all_images = grouped_df["image"].tolist()
    existing   = [im for im in all_images if os.path.exists(os.path.join(train_images_root, im))]
    if len(existing) < len(all_images):
        print(f"Warning: {len(set(all_images)-set(existing))} CSV images not found on disk. Proceeding with {len(existing)}.")
    train_names, val_names = train_test_split(existing, test_size=val_ratio, random_state=seed, shuffle=True)

    # Map per-image list of boxes
    box_map = {}
    for _, row in grouped_df.iterrows():
        img = row["image"]
        xs, ys, xe, ye = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        boxes = list(zip(xs, ys, xe, ye)) if isinstance(xs, list) else [(float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"]))]
        box_map[img] = boxes

    def write_yolo_label(img_src_path: str, img_name: str, out_label_path: str):
        img_w, img_h = read_image_size(img_src_path)
        lines = []
        for (xmin, ymin, xmax, ymax) in box_map.get(img_name, []):
            x_c, y_c, w, h = xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(out_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # copy train/val + labels
    for split, names, out_img_dir, out_lbl_dir in [
        ("train", train_names, DS["train_images"], DS["train_labels"]),
        ("val",   val_names,   DS["val_images"],   DS["val_labels"]),
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
    for t in list_images(test_images_root):
        safe_copy(t, os.path.join(DS["test_images"], os.path.basename(t)))

    # Write dataset.yaml (used by YOLOv8)
    yaml_text = f"""# Auto-generated dataset file
path: {normalize_path(WORKSPACE)}
train: {normalize_path(DS["train_images"])}
val:   {normalize_path(DS["val_images"])}
test:  {normalize_path(DS["test_images"])}
names: {json.dumps(CLASS_NAMES)}
nc: {NC}
"""
    with open(DS["yaml"], "w", encoding="utf-8") as f:
        f.write(yaml_text)

    return {"dataset_yaml": DS["yaml"]}


# ===============================================================
# Dataset + loader utilities for Torchvision/DETR training
# (reads the YOLO .txt labels we created)
# ===============================================================
class YoloTxtDataset(Dataset):
    """
    Reads images from <root_images> and YOLO txt labels from <root_labels>.
    Returns (image_tensor, target_dict).

    Key features:
    - Cross-platform label path construction (works on Windows/macOS/Linux).
    - Optionally filters out images that have no boxes (Torchvision requires >0).
    - Optionally shifts class ids by +1 for Torchvision (background at 0).
    - Returns tensors (C,H,W) so Torchvision models can consume directly.
    """

    def __init__(self, root_images: str, root_labels: str,
                 filter_empty: bool = False,
                 for_torchvision: bool = False):
        self.root_images = os.path.abspath(root_images)
        self.root_labels = os.path.abspath(root_labels)
        self.filter_empty = filter_empty
        self.for_torchvision = for_torchvision

        # List all images in the images root (flat or nested)
        self.images = self._list_images(self.root_images)

        # If desired, keep only images that actually have >=1 labeled box
        if self.filter_empty:
            keep = []
            for p in self.images:
                lp = self._label_path(p)
                if os.path.exists(lp):
                    with open(lp, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f.read().strip().splitlines() if ln.strip()]
                    # consider a line valid if it has the 5 YOLO fields
                    if any(len(ln.split()) == 5 for ln in lines):
                        keep.append(p)
            self.images = keep

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _list_images(root):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = []
        for base, _, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.join(base, fn))
        return sorted(paths)

    def _label_path(self, img_path: str) -> str:
        """
        Build the label path by taking the relative path of the image
        and swapping the extension for .txt under the labels root.
        This avoids fragile string replaces with forward/back slashes.
        """
        rel = os.path.relpath(img_path, self.root_images)  # e.g. "sub/abc.jpg"
        rel_no_ext = os.path.splitext(rel)[0]  # e.g. "sub/abc"
        return os.path.join(self.root_labels, rel_no_ext + ".txt")

    def _load_labels_xyxy(self, img_path: str, w: int, h: int):
        """
        Read YOLO normalized labels (class, xc, yc, w, h) and convert to
        absolute pixel xyxy boxes. Returns (boxes[N,4], classes[N]).
        """
        lbl_path = self._label_path(img_path)
        boxes, classes = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f.read().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, xc, yc, ww, hh = parts
                    cls_id = int(cls_id)
                    xc, yc, ww, hh = float(xc), float(yc), float(ww), float(hh)

                    # YOLO normalized -> absolute xyxy
                    x1 = (xc - ww / 2.0) * w
                    y1 = (yc - hh / 2.0) * h
                    x2 = (xc + ww / 2.0) * w
                    y2 = (yc + hh / 2.0) * h

                    # Clip to image bounds just in case
                    x1 = max(0.0, min(x1, w - 1))
                    y1 = max(0.0, min(y1, h - 1))
                    x2 = max(0.0, min(x2, w - 1))
                    y2 = max(0.0, min(y2, h - 1))

                    # Discard degenerate boxes (zero area)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)

        boxes = np.asarray(boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int64)

        # Torchvision models expect background at 0, so our single class "car" must be 1
        if self.for_torchvision and len(classes) > 0:
            classes = classes + 1  # 0->1 (car becomes label 1)

        return boxes, classes

    def __getitem__(self, idx):
        # Load and convert image to tensor (C,H,W) in [0,1]
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_tensor = TF.to_tensor(img)

        # Load labels for this image
        boxes, classes = self._load_labels_xyxy(img_path, w, h)

        # Build target dict that Torchvision expects
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),  # [N,4] xyxy
            "labels": torch.as_tensor(classes, dtype=torch.int64),  # [N]  in {1..num_classes-1}
            "image_id": torch.tensor([idx], dtype=torch.int64),
            # Optional fields used by some evaluation code:
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=torch.float32
            ) if len(boxes) else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return img_tensor, target


def collate_torchvision(batch):
    """
    Default collate for detection models: lists of images and lists of targets.
    """
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def collate_for_detr(processor: DetrImageProcessor):
    """
    Returns a collate_fn that:
    - takes (PIL image, target) items from YoloTxtDataset,
    - converts targets to COCO annotation style,
    - calls the processor to produce pixel_values + labels ready for DETR.
    """
    def _collate(batch):
        images = [b[0] for b in batch]
        annotations = []
        for img, tgt in batch:
            boxes_xyxy = tgt["boxes"].cpu().numpy().tolist()
            cats       = tgt["labels"].cpu().numpy().tolist()
            anns = []
            for b, c in zip(boxes_xyxy, cats):
                # DETR processor expects COCO bboxes: [x, y, w, h]
                x, y, x2, y2 = b
                anns.append({"bbox": [x, y, max(0.0, x2-x), max(0.0, y2-y)], "category_id": int(c)})
            annotations.append({"image_id": 0, "annotations": anns})
        processed = processor(images=images, annotations=annotations, return_tensors="pt")
        return processed
    return _collate

# ===============================================================
# Training: Torchvision (FasterRCNN / RetinaNet)
# ===============================================================
def build_torchvision_model(backbone: str, num_classes: int = 2):
    """
    Creates a Torchvision detection model with COCO-pretrained weights and sets num_classes.
    num_classes includes background (so for 1 class: use 2).
    """
    if backbone == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    elif backbone == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.backbone.out_channels
        model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.to(DEVICE)

@torch.no_grad()
def evaluate_torchvision(model, loader) -> float:
    model.eval()
    all_preds = []
    all_gts = {}
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        # accumulate GTs
        for t in targets:
            img_id = id(t)  # ephemeral id per batch (good enough for global bag)
            gts = t["boxes"].cpu().numpy().tolist()
            all_gts[img_id] = gts

        # accumulate predictions
        for out, t in zip(outputs, targets):
            img_id = id(t)
            boxes = out["boxes"].detach().cpu().numpy().tolist() if "boxes" in out else []
            scores = out["scores"].detach().cpu().numpy().tolist() if "scores" in out else []
            for b, s in zip(boxes, scores):
                all_preds.append((img_id, float(s), b))

    return compute_map50(all_preds, all_gts, iou_th=0.5)

def only_valid_images(images, targets):
    # Skip unlabeled samples in the batch
    valid_images, valid_targets = [], []
    for img, tgt in zip(images, targets):
        if len(tgt["boxes"]) > 0:
            valid_images.append(img)
            valid_targets.append(tgt)
        else:
            print("Skipped an unlabeled image.")

    return valid_images, valid_targets

# ===============================================================
# Training loop for Torchvision models (FasterRCNN / RetinaNet)
# ===============================================================
import torch
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_torchvision(backbone: str, run_name: str, epochs: int = EPOCHS_TV, batch_size: int = BATCH_TV, lr: float = LR_TV) -> Dict[str, str]:
    """
    Trains a Torchvision detection model (FasterRCNN/RetinaNet) on our YOLO-style dataset.
    Produces results.csv and saves best weights by val mAP@0.5.
    """
    # Only +1 for background if model requires it
    model = build_torchvision_model(backbone, num_classes=NC + 1)

    train_ds = YoloTxtDataset(DS["train_images"], DS["train_labels"])
    val_ds = YoloTxtDataset(DS["val_images"],   DS["val_labels"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_torchvision, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_torchvision, num_workers=NUM_WORKERS)

    # Optimizer from torchvision references
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    run_dir = os.path.join(WORKSPACE, "runs", run_name)
    ensure_dir(run_dir)
    results_csv = os.path.join(run_dir, "results.csv")
    weights_dir = os.path.join(run_dir, "weights")
    ensure_dir(weights_dir)

    best_map = -1.0
    best_path = os.path.join(weights_dir, "best.pth")
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            # Move to device
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Filter out invalid samples (no boxes)
            valid_pairs = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]
            if not valid_pairs:
                continue
            images, targets = zip(*valid_pairs)

            # Forward + loss
            loss_dict = model(list(images), list(targets))
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        # Validation (simple mAP@0.5)
        val_map = evaluate_torchvision(model, val_loader)

        # Log row and save results.csv
        row = {"epoch": epoch, "train/loss": epoch_loss / max(1, len(train_loader)), "val/mAP50": val_map}
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        # Save best weights
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_path)

        print(f"[{backbone}] Epoch {epoch:03d}/{epochs} | train_loss={row['train/loss']:.4f} | val_mAP50={val_map:.4f}")

    # Copy to ./models for convenience
    ensure_dir(MODELS_DIR)
    friendly = os.path.join(MODELS_DIR, f"{run_name}_best.pth")
    safe_copy(best_path, friendly)

    return {"run_dir": run_dir, "results_csv": results_csv, "best_weights": friendly}

# ===============================================================
# Training: Hugging Face DETR (fine-tune from COCO checkpoint)
# ===============================================================
def train_hf_detr(pretrained_id: str, run_name: str, epochs: int = EPOCHS_DETR, batch_size: int = BATCH_DETR, lr: float = LR_DETR) -> Dict[str, str]:
    """
    Fine-tunes DETR using the official HF processor and model.
    We build a data collator to feed pixel_values + labels.
    We log a results.csv (loss + val mAP@0.5) and save best weights.
    """
    processor = DetrImageProcessor.from_pretrained(pretrained_id)
    model = DetrForObjectDetection.from_pretrained(pretrained_id, num_labels=NC).to(DEVICE)

    # Datasets / loaders
    train_ds = YoloTxtDataset(DS["train_images"], DS["train_labels"])
    val_ds = YoloTxtDataset(DS["val_images"],   DS["val_labels"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_for_detr(processor), num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds,   batch_size=1,           shuffle=False, collate_fn=lambda x: x,             num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    run_dir = os.path.join(WORKSPACE, "runs", run_name)
    ensure_dir(run_dir)
    results_csv = os.path.join(run_dir, "results.csv")
    weights_dir = os.path.join(run_dir, "weights")
    ensure_dir(weights_dir)
    best_path = os.path.join(weights_dir, "best.pth")
    best_map = -1.0
    rows = []

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        # ---- Validation: build predictions for mAP50
        model.eval()
        all_preds = []
        all_gts = {}
        with torch.no_grad():
            for item in val_loader:
                # item is a list of length 1 because we batched with batch_size=1
                (img, tgt) = item[0]
                # keep original size for postprocess
                target_size = torch.tensor([img.size[::-1]]).to(DEVICE)  # (H,W)
                processed = processor(images=img, return_tensors="pt").to(DEVICE)
                outputs = model(**processed)
                # post-process to image space
                results = processor.post_process_object_detection(outputs, target_sizes=target_size)[0]
                boxes = results["boxes"].detach().cpu().numpy().tolist()
                scores = results["scores"].detach().cpu().numpy().tolist()

                # accumulate GT
                img_id = id(tgt)  # unique per-sample
                all_gts[img_id] = tgt["boxes"].cpu().numpy().tolist()

                # accumulate predictions
                for b, s in zip(boxes, scores):
                    all_preds.append((img_id, float(s), b))

        val_map = compute_map50(all_preds, all_gts, iou_th=0.5)

        row = {"epoch": epoch, "train/loss": epoch_loss/len(train_loader), "val/mAP50": val_map}
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_path)

        print(f"[DETR] Epoch {epoch:03d}/{epochs} | train_loss={row['train/loss']:.4f} | val_mAP50={val_map:.4f}")

    # Friendly copy
    ensure_dir(MODELS_DIR)
    friendly = os.path.join(MODELS_DIR, f"{run_name}_best.pth")
    safe_copy(best_path, friendly)

    return {"run_dir": run_dir, "results_csv": results_csv, "best_weights": friendly}

# ===============================================================
# Training: Ultralytics YOLO (unchanged from your previous flow)
# ===============================================================
def get_or_download_pretrained(pretrained_name: str, local_dir: str) -> str:
    ensure_dir(local_dir)
    local_path = os.path.join(local_dir, pretrained_name)
    if os.path.exists(local_path):
        print(f"Using local pretrained weights: {local_path}")
        return local_path
    print(f"Downloading pretrained weights via Ultralytics for: {pretrained_name}")
    y = YOLO(pretrained_name)
    try:
        src_candidates = []
        if hasattr(y, "ckpt_path") and y.ckpt_path:
            src_candidates.append(y.ckpt_path)
        if hasattr(y, "model") and hasattr(y.model, "pt_path") and y.model.pt_path:
            src_candidates.append(y.model.pt_path)
        for c in src_candidates:
            if c and os.path.exists(c):
                safe_copy(c, local_path)
                break
    except Exception:
        pass
    return local_path if os.path.exists(local_path) else pretrained_name

def train_yolo(model_label: str, pretrained: str, run_name: str, dataset_yaml: str) -> Dict[str, str]:
    print(f"\n=== Training {model_label} (Ultralytics) ===")
    weights = get_or_download_pretrained(pretrained, MODELS_DIR)
    model = YOLO(weights)
    _ = model.train(
        data=dataset_yaml,
        epochs=EPOCHS_YOLO,
        imgsz=IMGSZ,
        batch=16,
        project=os.path.join(WORKSPACE, "runs"),
        name=run_name,
        verbose=True,
        exist_ok=True
    )
    run_dir = os.path.join(WORKSPACE, "runs", "detect", run_name)
    results_csv = os.path.join(run_dir, "results.csv")
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    ensure_dir(MODELS_DIR)
    dst_best = os.path.join(MODELS_DIR, f"{model_label}_best.pt")
    if os.path.exists(best_weights):
        safe_copy(best_weights, dst_best)
        print(f"Saved best YOLO weights to: {dst_best}")
    return {"run_dir": run_dir, "results_csv": results_csv, "best_weights": dst_best if os.path.exists(dst_best) else best_weights}

# ===============================================================
# Inference (write test_predictions_<MODEL>.csv)
# ===============================================================
@torch.no_grad()
def predict_torchvision_csv(model, images_dir: str, out_csv: str):
    model.eval()
    rows = []
    for p in list_images(images_dir):
        im = Image.open(p).convert("RGB")
        inp = TVF.to_tensor(im).unsqueeze(0).to(DEVICE)
        outputs = model(inp)[0]
        boxes = outputs.get("boxes", torch.empty(0,4)).detach().cpu().numpy()
        for (xmin, ymin, xmax, ymax) in boxes:
            rows.append([os.path.basename(p), float(xmin), float(ymin), float(xmax), float(ymax)])
    pd.DataFrame(rows, columns=["image","xmin","ymin","xmax","ymax"]).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

@torch.no_grad()
def predict_detr_csv(processor, model, images_dir: str, out_csv: str):
    model.eval()
    rows = []
    for p in list_images(images_dir):
        im = Image.open(p).convert("RGB")
        size = torch.tensor([im.size[::-1]]).to(DEVICE)  # (H,W)
        inputs = processor(images=im, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        res = processor.post_process_object_detection(outputs, target_sizes=size)[0]
        boxes = res["boxes"].detach().cpu().numpy()
        for (xmin, ymin, xmax, ymax) in boxes:
            rows.append([os.path.basename(p), float(xmin), float(ymin), float(xmax), float(ymax)])
    pd.DataFrame(rows, columns=["image","xmin","ymin","xmax","ymax"]).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

def predict_yolo_csv(weights_path: str, images_dir: str, out_csv: str):
    model = YOLO(normalize_path(weights_path))
    rows = []
    results = model.predict(source=normalize_path(images_dir), imgsz=IMGSZ, stream=True, verbose=False)
    for r in results:
        im_name = os.path.basename(getattr(r, "path", ""))
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        for (xmin, ymin, xmax, ymax) in xyxy:
            rows.append([im_name, float(xmin), float(ymin), float(xmax), float(ymax)])
    pd.DataFrame(rows, columns=["image","xmin","ymin","xmax","ymax"]).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

# ===============================================================
# Report (curves + sample detections)
# ===============================================================
def read_results_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

def sample_detection_grid_predict(fn_predict_one, images_dir: str, n: int = 12, seed: int = 123):
    random.seed(seed)
    imgs = list_images(images_dir)
    if not imgs:
        fig = plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No images to visualize", ha='center', va='center')
        plt.axis('off'); return fig
    picks = random.sample(imgs, min(n, len(imgs)))
    cols = 4; rows = math.ceil(len(picks)/cols)
    fig = plt.figure(figsize=(18, 4*rows))
    for i, p in enumerate(picks):
        ax = fig.add_subplot(rows, cols, i+1)
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ax.imshow(img); ax.set_title(os.path.basename(p)); ax.axis('off')
        for (xmin, ymin, xmax, ymax) in fn_predict_one(p):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, linewidth=2, edgecolor="red"))
    plt.tight_layout()
    return fig

def plot_training_curves(df: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1,1,1)
    if "train/loss" in df.columns:
        ax.plot(df["epoch"], df["train/loss"], label="train/loss")
    if "val/mAP50" in df.columns:
        ax.plot(df["epoch"], df["val/mAP50"], label="val/mAP50")
    # Ultralytics CSV may have its own column names; draw some if present
    for col, nice in [
        ("metrics/mAP50(B)", "mAP50(B)"),
        ("metrics/mAP50-95(B)", "mAP50-95(B)"),
        ("train/box_loss", "YOLO train box loss"),
        ("val/box_loss", "YOLO val box loss"),
    ]:
        if col in df.columns: ax.plot(df.index, df[col], label=nice)
    ax.set_title(title); ax.set_xlabel("Epoch/Step"); ax.set_ylabel("Value"); ax.legend()
    plt.tight_layout(); return fig

def generate_pdf_report(out_pdf: str):
    with PdfPages(out_pdf) as pdf:
        # Title
        fig = plt.figure(figsize=(11.7, 8.3))
        plt.text(0.5, 0.8, "Car Detection — Training Report", ha='center', va='center', fontsize=22, weight='bold')
        lines = [
            "Models: YOLOv8 (s, n, scratch), FasterRCNN, RetinaNet, DETR",
            "Dataset: Your CSV → YOLOv8 structure",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Each section: training curves + sample detections"
        ]
        for i, line in enumerate(lines):
            plt.text(0.5, 0.6 - 0.06*i, line, ha='center', va='center', fontsize=12)
        plt.axis('off'); pdf.savefig(fig); plt.close(fig)

        # Per-model pages
        for cfg in MODEL_CONFIGS:
            label = cfg["name"]; art = per_model.get(label, {})
            # Curves
            df = read_results_csv(art.get("results_csv",""))
            fig = plot_training_curves(df, f"{label} — Training Curves")
            pdf.savefig(fig); plt.close(fig)

            # Samples
            vis_dir = DS["val_images"] if list_images(DS["val_images"]) else DS["test_images"]

            if cfg["type"] == "ultralytics":
                model = YOLO(art["best_weights"])
                def _pred_one(pth):
                    r = model.predict(source=pth, imgsz=IMGSZ, verbose=False)[0]
                    if r.boxes is None or len(r.boxes)==0: return []
                    return r.boxes.xyxy.cpu().numpy().tolist()
                fig = sample_detection_grid_predict(_pred_one, vis_dir, n=12)
            elif cfg["type"] == "torchvision":
                m = build_torchvision_model(cfg["backbone"], num_classes=NC+1)
                m.load_state_dict(torch.load(art["best_weights"], map_location=DEVICE)); m.eval().to(DEVICE)
                def _pred_one(pth):
                    im = Image.open(pth).convert("RGB")
                    out = m(TVF.to_tensor(im).unsqueeze(0).to(DEVICE))[0]
                    return out.get("boxes", torch.empty(0,4)).detach().cpu().numpy().tolist()
                fig = sample_detection_grid_predict(_pred_one, vis_dir, n=12)
            else:  # DETR
                processor = DetrImageProcessor.from_pretrained(cfg["pretrained_id"])
                m = DetrForObjectDetection.from_pretrained(cfg["pretrained_id"], num_labels=NC).to(DEVICE)
                m.load_state_dict(torch.load(art["best_weights"], map_location=DEVICE)); m.eval()
                def _pred_one(pth):
                    im = Image.open(pth).convert("RGB")
                    size = torch.tensor([im.size[::-1]]).to(DEVICE)
                    inp = processor(images=im, return_tensors="pt").to(DEVICE)
                    out = m(**inp)
                    res = processor.post_process_object_detection(out, target_sizes=size)[0]
                    return res["boxes"].detach().cpu().numpy().tolist()
                fig = sample_detection_grid_predict(_pred_one, vis_dir, n=12)

            fig.suptitle(f"{label} — Sample Detections", fontsize=14, y=1.02)
            pdf.savefig(fig); plt.close(fig)

    print(f"Saved report: {out_pdf}")

# ===============================================================
# Orchestration: train() + test()
# ===============================================================
def train_all():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    # Sanity checks
    ensure_dir_with_msg(TRAIN_IMAGES_DIR, f"Expected training images at {TRAIN_IMAGES_DIR}")
    ensure_dir_with_msg(TEST_IMAGES_DIR,  f"Expected testing images at {TEST_IMAGES_DIR}")
    ensure_dir_with_msg(TRAIN_CSV,       f"Expected CSV at {TRAIN_CSV}")
    ensure_dir(WORKSPACE); ensure_dir(MODELS_DIR)

    print("Loading training CSV and preparing dataset ...")
    grouped = load_and_group_training_csv(TRAIN_CSV)
    _ = build_yolo_dataset_structure(grouped, TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, WORKSPACE, VAL_SPLIT, RANDOM_SEED)

    # Train per model
    for cfg in MODEL_CONFIGS:
        if cfg["type"] == "ultralytics":
            art = train_yolo(cfg["name"], cfg["pretrained"], cfg["run_name"], DS["yaml"])
        elif cfg["type"] == "torchvision":
            art = train_torchvision(cfg["backbone"], cfg["run_name"], epochs=EPOCHS_TV, batch_size=BATCH_TV, lr=LR_TV)
        else:  # hf_detr
            art = train_hf_detr(cfg["pretrained_id"], cfg["run_name"], epochs=EPOCHS_DETR, batch_size=BATCH_DETR, lr=LR_DETR)
        per_model[cfg["name"]] = art

def test_all():
    ensure_dir_with_msg(DS["test_images"], f"Expected testing images at {DS['test_images']}")

    # Run inference per model and write CSVs
    for cfg in MODEL_CONFIGS:
        label = cfg["name"]
        out_csv = f"test_predictions_{label}.csv"
        art = per_model[label]

        if cfg["type"] == "ultralytics":
            predict_yolo_csv(art["best_weights"], DS["test_images"], out_csv)

        elif cfg["type"] == "torchvision":
            model = build_torchvision_model(cfg["backbone"], num_classes=NC+1)
            model.load_state_dict(torch.load(art["best_weights"], map_location=DEVICE))
            model = model.to(DEVICE).eval()
            predict_torchvision_csv(model, DS["test_images"], out_csv)

        else:  # hf_detr
            processor = DetrImageProcessor.from_pretrained(cfg["pretrained_id"])
            model = DetrForObjectDetection.from_pretrained(cfg["pretrained_id"], num_labels=NC).to(DEVICE)
            model.load_state_dict(torch.load(art["best_weights"], map_location=DEVICE))
            predict_detr_csv(processor, model, DS["test_images"], out_csv)

    # Report
    generate_pdf_report(REPORT_PDF)

def main():
    train_all()      # comment out if you only want to run test on already-trained models
    test_all()

if __name__ == "__main__":
    main()
