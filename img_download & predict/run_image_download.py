from Image_Path_and_Download import (
    generate_image_urls_sagehen,
    generate_image_urls_sagehen2,
    generate_image_urls_sagehen3,
    download_urls_to_folder_and_metadata
)

from pathlib import Path 

# === USER SETTINGS ===
SITES = ["sagehen", "sagehen2", "sagehen3"]
NUM_IMAGES = 20000

# === EXECUTE FOR EACH SITE ===
for SITE_NAME in SITES:
    SAVE_FOLDER = Path(f"/home/nmichelotti/Desktop/Senior Model/Site Data/images_{SITE_NAME}")
    print(f"\nFetching up to {NUM_IMAGES} images from site: {SITE_NAME}")

    if SITE_NAME == "sagehen":
        urls = generate_image_urls_sagehen(SITE_NAME, limit=NUM_IMAGES)
    elif SITE_NAME == "sagehen2":
        urls = generate_image_urls_sagehen2(SITE_NAME, limit=NUM_IMAGES)
    elif SITE_NAME == "sagehen3":
        urls = generate_image_urls_sagehen3(SITE_NAME, limit=NUM_IMAGES)
    else:
        print(f"Skipping unknown site: {SITE_NAME}")
        continue

    if not urls:
        print("\nNo image URLs were successfully generated.")
    else:
        print("\nDownloading images and metadata...")
        download_urls_to_folder_and_metadata(urls, SAVE_FOLDER)
        print("\nDone.")
