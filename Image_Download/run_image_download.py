from Image_Path_and_Download import _generate_image_urls, download_urls_to_folder_and_metadata
from pathlib import Path

# === USER SETTINGS ===
SAVE_ROOT = Path("/home/nmichelotti/Desktop/Senior Model/test_data")

SITE_CONFIGS = [
    {
        "site_name": "sagehen", 
        "num_images": 2, # How many Images to download
        "start_year": 2017,
        "start_month": 10, # Jan = 1, Dec = 12
        "start_day": 5, 
        "valid_minutes": [56, 26] #Minutes When Photos Are Taken
    },

    # Example of another site config:
    # {
    #     "site_name": "sagehen2",
    #     "num_images": 3,
    #     "start_year": 2018,
    #     "start_month": 6,
    #     "start_day": 10,
    #     "valid_minutes": [28, 58]
    # },
]

# === MAIN LOOP ===
for config in SITE_CONFIGS:
    site = config["site_name"]
    save_folder = SAVE_ROOT / site
    print(f"\nFetching {config['num_images']} images from {site}")

    urls = _generate_image_urls(
        site_name=site,
        limit=config["num_images"],
        start_year=config["start_year"],
        start_month=config["start_month"],
        start_day=config["start_day"],
        valid_minutes=config["valid_minutes"],
        save_to=save_folder
    )

    if not urls:
        print("No valid image URLs found.")
    else:
        print("Downloading images and metadata...")
        download_urls_to_folder_and_metadata(urls, save_folder)
        print("✓ Done.")
