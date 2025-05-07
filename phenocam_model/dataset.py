import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Dataset Class
class PhenoCamSequenceDataset(Dataset):
    def __init__(self, image_dir, metadata_csvs, meta_features, sequence_length=90, predict_ahead_days=14):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image_dir = image_dir
        self.meta_features = meta_features
        self.sequence_length = sequence_length
        self.predict_ahead_days = predict_ahead_days
        self.samples = []

        all_metadata = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            all_metadata.append(df)
        self.full_df = pd.concat(all_metadata, ignore_index=True)
        self.full_df = self.full_df.sort_values("date")
        self.full_df = self.full_df.reset_index(drop=True)

        for i in tqdm(range(len(self.full_df) - self.sequence_length - self.predict_ahead_days), desc="Building sequences"):
            seq = self.full_df.iloc[i:i+self.sequence_length]
            target_row = self.full_df.iloc[i+self.sequence_length+self.predict_ahead_days-1]
            self.samples.append((seq, target_row))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target_row = self.samples[idx]

        imgs = []
        for _, row in seq.iterrows():
            filename = str(row.get('image_filename') or row.get('filename')).strip()
            img_path = self.image_dir / filename
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
            else:
                img = torch.zeros((3, 224, 224))
            imgs.append(img)
        imgs = torch.stack(imgs)

        metas = []
        for _, row in seq.iterrows():
            meta_features = []
            for col in self.meta_features:
                val = row.get(col, 0.0)
                meta_features.append(float(val) if pd.notna(val) else 0.0)
            metas.append(meta_features)
        metas = torch.tensor(metas, dtype=torch.float32)

        label = int(target_row.get("snow_flag", 0)) if not pd.isna(target_row.get("snow_flag")) else 0

        return imgs, metas, torch.tensor(label, dtype=torch.long)
