import gdown
import zipfile
import os

file_id = "1k4UuaUJr8F39rxXRloGEJCSOxs_z2VW8"
url = f"https://drive.google.com/uc?id={file_id}&export=download"
zip_filename = "yolov8n.zip"

print("Downloading ZIP from Google Drive...")
gdown.download(url, output=zip_filename, quiet=False, fuzzy=True)

print("Extracting ZIP...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())

print("Cleaning up...")
os.remove(zip_filename)

print("All files extracted into the current folder.")