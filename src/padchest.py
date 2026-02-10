import os
import gdown

# URL of the Google Drive folder
folder_url = "https://drive.google.com/drive/folders/1tthDMG019wk4Oq4ar6EHAy20-xl1cVra?usp=sharing"

# Destination path for the download
destination_path = "/gpfs/workdir/restrepoda/datasets/PADCHEST"

# Ensure the destination directory exists
os.makedirs(destination_path, exist_ok=True)

# Download the folder to the specified destination path
gdown.download_folder(url=folder_url, output=destination_path, quiet=False)
