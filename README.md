# PhenoCam Image Downloader and Weather Prediction

This repository provides a complete pipeline for:

1. Downloading PhenoCam images and associated metadata  
2. *(Optional)* Training a custom deep learning model using the downloaded image data  
3. Running weather and snow predictions using a multi-task deep learning model  

---

## Configuration: Downloading Images

To begin downloading PhenoCam images, open the `run_image_download.py` script in the `image_download` folder and edit the user-defined settings.

### Required Parameters

- **`SAVE_ROOT`**  
  The root directory where all images and metadata will be saved.

- **`SITE_CONFIGS`**  
  A list of dictionaries—each corresponding to one site—with the following fields:

  - `site_name`: The PhenoCam site ID (e.g., `"sagehen2"`) found on the [PhenoCam website](https://phenocam.nau.edu/webcam/).
  - `num_images`: The number of images to download from this site.
  - `start_year`, `start_month`, `start_day`: The starting date for the image query.
  - `valid_minutes`: A list of image capture minutes to match (e.g., `[0, 30]`).  
    This ensures only valid timestamps are considered. PhenoCam typically captures two images per hour at consistent minute marks. You should inspect recent image timestamps from the site and specify the most common capture minutes here for more efficient URL generation.

You can add as many site dictionaries to `SITE_CONFIGS` as needed—just copy and customize the format for each additional site.

### Output Directory Structure

Images and metadata for each site will be saved in a subdirectory under your specified `SAVE_ROOT`. The structure will look like this:

- /your/save/path/
  - site_name/
    - image1.jpg
    - image2.jpg
    - ...
    - metadata.csv

---

## Training Deep Learning Model


---
## Generating Weather Reports
