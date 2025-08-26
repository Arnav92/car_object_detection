import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=1tQjfut82kGNj2gN1iALNfY2oxVlkc9l9"
zip_filename = "car_detection_report.zip"

print("Downloading ZIP from Google Drive...")
gdown.download(url, output=zip_filename, quiet=False, fuzzy=True)

print("Extracting ZIP...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())

print("Cleaning up...")
os.remove(zip_filename)

print("All files extracted into the current folder.")