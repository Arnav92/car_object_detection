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

If the script finishes with the message:

```
All files extracted into the current folder!
```

## Step 2: If Automatic Download Fails

If `setup.py` fails for any reason:

1. Go to the Google Drive link directly:  
   https://drive.google.com/drive/folders/1hOLUIcf3wY-vp1xabdzFR5pqEj7Nt4ru?hl=de

2. Download and extract the zip.

3. Place the files into the same folder as this README.

## Step 3: Alternative Download Source

If Google Drive doesn't work, you can download and extract from here:  
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

After setup, your project should contain the following key components:

### `data/` folder
Contains all images and bounding boxes for training and testing, including histograms of data analysis. Notably, `personal_images/` contains all images taken in person and tested on the models.

### `workspace/` folder
Where all data and models are organized. The `test/`, `train/`, and `val/` subdirectories are self-explanatory. The `runs/` folder contains important details about all 3 models, including:
- Model weights
- Box curves
- Confusion matrices
- Raw training results data
- Example images from training batches

### Root directory files
- **`car_detection_report.pdf`** - Summarizes training results and provides sample test images with bounding boxes from all models
- **`personal_images_report.pdf`** - Results of testing the models on all images in `./data/personal_images/`
- **`histogram_maker.py`** - Script that generated the histograms in `./data/data_description_histograms.pdf`
- **`main.py`** - Main training and testing script that generates the above reports
- **`setup.py`** - Downloads and sets up all required files
- **`speed_metrics.json`** - Contains performance speed information for all models, generated in `main.py` 
- **`.csv files`** - Raw bounding box prediction values from all models