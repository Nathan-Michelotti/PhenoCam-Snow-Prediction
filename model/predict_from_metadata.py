import os
import torch
import pandas as pd
from pathlib import Path
from model import PhenoCamResNet, META_FEATURES  # Ensure this matches your actual import
from torchvision import models

# Configuration
CSV_PATH = Path("/home/nmichelotti/Desktop/Senior Model/Site Data/future_avg_metadata.csv")
MODEL_PATH = Path("/home/nmichelotti/Desktop/Senior Model/model/PhenoCam_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metadata
df = pd.read_csv(CSV_PATH)
print(f"Loaded metadata CSV with shape: {df.shape}")

# Drop rows with any missing values in META_FEATURES
df = df.dropna(subset=META_FEATURES)
meta_tensor = torch.tensor(df[META_FEATURES].values, dtype=torch.float32).to(DEVICE)

# Create dummy image tensor (not used for future prediction)
dummy_img = torch.zeros((len(meta_tensor), 3, 224, 224), dtype=torch.float32).to(DEVICE)

# Load model
meta_dim = len(META_FEATURES)
model = PhenoCamResNet(resnet="resnet18", n_classes=2, meta_dim=meta_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Run prediction
with torch.no_grad():
    outputs = model(dummy_img, meta_tensor)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    confidences = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Confidence for class 1 (snow)

# Save predictions
df['PredictedSnow'] = predictions
df['SnowConfidence'] = confidences
output_path = CSV_PATH.parent / "future_predictions.csv"
df.to_csv(output_path, index=False)
print(f"Saved predictions to: {output_path}")
