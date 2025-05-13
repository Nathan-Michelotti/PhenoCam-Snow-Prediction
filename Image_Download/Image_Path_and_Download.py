from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
from pathlib import Path
import pandas as pd
import random
import requests
from tqdm import tqdm


def specific_site(site_name, limit=5, save_to=None):
    return _generate_image_urls(site_name, limit, start_year=2017, start_month=10, start_day=5, valid_minutes=[56, 26], save_to=save_to)


def _generate_image_urls(site_name, limit, start_year, start_month, start_day, valid_minutes, save_to=None):
    base_url = "https://phenocam.nau.edu/data/archive"
    verified_urls = set()

    # Read existing URLs if they exist
    if save_to:
        url_file = Path(save_to) / f"{site_name}_urls.txt"
        if url_file.exists():
            with open(url_file, "r") as f:
                for line in f:
                    verified_urls.add(line.strip())

    end_year = 2025
    start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    days = list(pd.date_range(start=start_date, end=f"{end_year}-12-31", freq="D"))

    attempts = 0
    max_attempts = 100000

    from tqdm import tqdm
    with tqdm(total=limit, desc=f"Verifying image URLs from {site_name}") as pbar:
        while len(verified_urls) < limit and attempts < max_attempts:
            day = random.choice(days)
            h = random.randint(0, 23)
            m = random.choice(valid_minutes)

            year = day.strftime("%Y")
            month = day.strftime("%m")
            day_str = day.strftime("%d")

            timestamp = f"{str(h).zfill(2)}{str(m).zfill(2)}05"
            img_name = f"{site_name}_{year}_{month}_{day_str}_{timestamp}.jpg"
            img_url = f"{base_url}/{site_name}/{year}/{month}/{img_name}"

            if img_url in verified_urls:
                attempts += 1
                continue

            try:
                head = requests.head(img_url, timeout=3)
                if head.ok:
                    verified_urls.add(img_url)
                    pbar.update(1)
            except:
                pass

            attempts += 1

    return list(verified_urls)[:limit]



def download_direct_image_and_meta(url, save_to: Path, metadata_rows: list, skip_existing=True):
    filename = url.split("/")[-1]
    base = filename.replace(".jpg", "")
    parts = base.split("_")
    site = parts[0]
    date = f"{parts[1]}-{parts[2]}-{parts[3]}"
    time = parts[4]

    img_path = save_to / f"{base}.jpg"
    meta_url = url.replace(".jpg", ".meta")
    meta_data = {}

    # Skip if image already exists
    if skip_existing and img_path.exists():
        return

    # Download image
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok:
            img = Image.open(BytesIO(resp.content))
            img.save(img_path)
        else:
            print(f"Failed image: {url}")
            return
    except Exception as e:
        print(f"Image error {url}: {e}")
        return

    # Download metadata
    try:
        resp = requests.get(meta_url, timeout=10)
        if resp.ok:
            for line in resp.text.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    meta_data[key.strip()] = value.strip()
        else:
            print(f"Failed metadata: {meta_url}")
    except Exception as e:
        print(f"Metadata error {meta_url}: {e}")

    # Combine metadata with image info
    meta_data.update({
        "image_filename": f"{base}.jpg",
        "image_url": url,
        "site": site,
        "date": date,
        "time": time
    })

    metadata_rows.append(meta_data)


def download_urls_to_folder_and_metadata(urls, save_to: Path):
    save_to.mkdir(parents=True, exist_ok=True)

    # Load existing URL list
    existing_urls = set()
    url_file = save_to / f"{urls[0].split('/')[5]}_urls.txt"
    if url_file.exists():
        with open(url_file, "r") as f:
            existing_urls = set(line.strip() for line in f)

    # Filter out already downloaded URLs
    urls = [url for url in urls if url not in existing_urls]

    # Load existing metadata
    metadata_path = save_to / "metadata.csv"
    if metadata_path.exists():
        existing_df = pd.read_csv(metadata_path)
        metadata_rows = existing_df.to_dict(orient="records")
    else:
        metadata_rows = []

    # Download new images + metadata
    for url in tqdm(urls, desc="Downloading images and metadata"):
        download_direct_image_and_meta(url, save_to, metadata_rows)

    # Save updated metadata and URL list
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.to_csv(metadata_path, index=False)

        with open(url_file, "w") as f:
            for row in metadata_rows:
                f.write(row["image_url"] + "\n")
