import os
from collections import Counter
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# === DARKNESS FILTER FUNCTION ===
def is_too_dark(image_path, threshold=15, darkness_ratio=0.95):
    try:
        img = Image.open(image_path).convert("L")
        pixels = np.array(img)
        dark_pixels = (pixels < threshold).sum()
        return (dark_pixels / pixels.size) > darkness_ratio
    except Exception as e:
        print(f"Error checking darkness for {image_path}: {e}")
        return False

# === DATASET CLASS ===
class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths, self.dark, self.weather, self.snow = [], [], [], []
        self.transform = transform

        label_map = {
            "Sunny_No_Snow": (0, 0, 0),
            "Sunny_With_Snow": (0, 0, 1),
            "Cloudy_No_Snow": (0, 1, 0),
            "Cloudy_With_Snow": (0, 1, 1),
            "to_dark": (1, 0, 0)
        }

        for folder, (d, w, s) in label_map.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(folder_path, f)
                    if folder == "to_dark" or not is_too_dark(full_path):
                        self.paths.append(full_path)
                        self.dark.append(d)
                        self.weather.append(w)
                        self.snow.append(s)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raise NotImplementedError("Image loading not needed for label distribution.")

# === MAIN ===
if __name__ == "__main__":
    transform = transforms.Compose([])  # Placeholder
    dataset = WeatherDataset("/home/nmichelotti/Desktop/data", transform)

    print(f"\nDataset size: {len(dataset)} samples")
    print("Label distribution:")
    print(" - Too Dark:", Counter(dataset.dark))
    print(" - Weather (0=Sunny, 1=Cloudy):", Counter(dataset.weather))
    print(" - Snow (0=No Snow, 1=Snow):", Counter(dataset.snow))
