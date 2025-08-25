import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paths
DATA_DIR = os.path.join("data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_solution_bounding_boxes (1).csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")

# Load bounding box CSV
df = pd.read_csv(TRAIN_CSV)

# PDF output path
output_pdf = os.path.join(DATA_DIR, "data_description_histograms.pdf")
pdf = PdfPages(output_pdf)

# --- 1. Histogram of bounding box areas ---
df["width"] = df["xmax"] - df["xmin"]
df["height"] = df["ymax"] - df["ymin"]
df["area"] = (df["width"] * df["height"]) / 256880

avg_area = df["area"].mean()

plt.figure(figsize=(8, 5))
plt.hist(df["area"], bins=50, color="skyblue", edgecolor="black")
plt.title(f"Histogram of Bounding Box Areas (Average Area = {avg_area:.2f})")
plt.xlabel("Bounding Box Area (pixels^2)")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.6)
pdf.savefig()   # save current figure to PDF
plt.close()

# --- 2. Histogram of object count per image ---
box_counts = df.groupby("image").size()

# Count all images in folder
all_images = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
all_images_set = set(all_images)

# Include images with zero boxes
'''image_box_count = {img: 0 for img in all_images_set}
image_box_count.update(box_counts.to_dict())

image_box_count_series = pd.Series(image_box_count)''' # <-- Delete below line and uncomment this to include 0 cars
image_box_count_series = df.groupby("image").size()
avg_boxes = image_box_count_series.mean()

plt.figure(figsize=(8, 5))
plt.hist(image_box_count_series, bins=range(0, image_box_count_series.max() + 2),
         align="left", color="salmon", edgecolor="black", rwidth=0.85)
plt.title(f"Histogram of Object Count per Image (Average = {avg_boxes:.2f})")
plt.xlabel("Number of Cars per Image")
plt.ylabel("Count of Images")
plt.grid(True, linestyle="--", alpha=0.6)
pdf.savefig()
plt.close()

# Close PDF file
pdf.close()

print(f"✅ Saved histograms to {output_pdf}")