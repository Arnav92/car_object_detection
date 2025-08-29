# Project Setup and Instructions

## Step 1: Get the Data and Files (VERY IMPORTANT)

Before running anything else, you need to ensure that **all required data, models, and other relevant files** are in your local folder.

First, install the required dependency if you haven't already:

```bash
pip install gdown
```

> **Note:** If you get a library error, you can also just click the download button when hovering over the error message.

The easiest way to get all files is by running:

```bash
python setup.py
```

> **Note:** Running `setup.py` can take several minutes to complete.

The script was a success if the console outputs:

```
All files extracted into the current folder!
```

## Step 2: If Automatic Download Fails

If `setup.py` fails for any reason:

1. Go to the Google Drive link directly (download "yolov8n.zip"):  
   https://drive.google.com/drive/folders/1hOLUIcf3wY-vp1xabdzFR5pqEj7Nt4ru?hl=de

2. Download and extract the zip.

3. Place the files into the same folder as this README.

Or pull this Github repository:
https://github.com/Arnav92/car_object_detection

## Step 3: Alternative Download Source

If Google Drive, nor Github works, you can download and extract from here:  
https://syncandshare.lrz.de/getlink/fiT88vDosahwxVPBmVtDzk/

**Important:** Make sure that all files downloaded are placed exactly as they appear in the zip file structure in your README file location.

## Step 4: Absolute Worst-Case Scenario

If none of the above steps work:

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/sshikamaru/car-object-detection

2. Open `main.py` and uncomment the call to `train()` inside the `main()` function.

3. Check the variable `DATA_DIR` at the top of the code:
   - Read the comments carefully.
   - Make sure your dataset folder is structured as expected.

> **Warning:** We strongly discourage using this method to collect all files. This can take several hours to complete as you would be training all models. Please make sure you have at least tried everything from above before resorting to this!

## Important Cleanup

If you go with the Kaggle fallback dataset, you will also need to remove the following lines from the code, as `PERSONAL_DIR` will not exist:

```python
personal_report = generate_personal_images_report(
    per_model,
    out_pdf="personal_images_report.pdf",
    personal_images_dir=PERSONAL_DIR
)
```

Also remove the related print statement that would otherwise raise an error.

## Ready to Run!

Once you have the data set up and the folder structure looks correct, you have all of the files relevant to our project!

## File Structure Overview

After setup, your project contains the following key components.

### `data/` folder
- Contains all images and bounding boxes for training and testing.
- `personal_images/` — images taken locally and used to test the models.
- `extra.pdf` — consolidated data-analysis PDF. This file includes various graphs such as histograms, and an example image showing a class instance with the **average-sized** bounding box.
- `regression_report.pdf` — for every permutation of models and for every metric in `results.csv`, this report shows:
  - regression plots of the metric vs. epoch for each model,
  - the difference in metric values between models,
  - statistical tests/conclusions on whether any variable is significant.

### `models/` folder
- Contains downloaded model weight files (e.g. `.pt`) for the YOLO and FastYOLO variants used.
- All provided model weights are **pretrained on the MS COCO dataset** (these are the COCO-pretrained checkpoints used as starting points).

### `workspace/` folder
- Organization used during training and evaluation. Typical structure:
  - `train/`, `val/`, `test/` — dataset splits used for training and evaluation.
  - `runs/` — per-run outputs (model weights, training logs, loss/metric curves, confusion matrices, sample training/validation images, etc.).

### Root directory files
- **`car_detection_report.pdf`** — main report summarizing training results, sample test images, and primary figures.
- **`personal_images_report.pdf`** — results of testing models on `./data/personal_images/`.
- **`extra.pdf`** — replaced `histogram_maker.pdf`; contains extra graphs referenced from the report (kept in the appendix).
- **`extra.py`** — script that generates `extra.pdf` and `regression_report.pdf`.
- **`main.py`** — main training / evaluation script that produces reports and raw results.
- **`setup.py`** — download/setup helper (models, required files, etc.).
- **`speed_metrics.json`** — detailed performance data: stores **per-image latency values** (for validation and test images for every model) and derived summary metrics such as `throughput_fps`, median latency, and other aggregated statistics.
- **`*.csv`** files — raw detection / bounding-box prediction outputs and `results.csv` (summary metrics used to build `regression_report.pdf`).
- **`*.pt`** files — model weight files (COCO-pretrained checkpoints and any locally saved checkpoints).

> **Note:** Many images in the .pdf files were not used in the report due to the page limit, but can still provide useful insights!