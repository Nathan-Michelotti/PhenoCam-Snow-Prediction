import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

from model import PhenoCamLSTM  # <-- IMPORT model cleanly here!

# Paths and settings
MODEL_PATH = "PhenoCam_LSTM_model.pth"
IMAGE_FOLDER = Path("../Site Data/images_sagehen")  # <- corrected relative path
CSV_PATH = Path("../data_orginization/sagehen/3_sagehen_EN_1000_roistats.csv")  # <- corrected
SEQUENCE_LENGTH = 90
META_FEATURES = [
    # (same metadata feature list you had)
    'doy', 'solar_elev', 'exposure', 'exposure_rgb', 'exposure_ir', 'awbflag', 'mask_index',
    'gcc', 'rcc', 'r_mean', 'g_mean', 'b_mean', 'ir_mean', 'ir_std',
    'r_std', 'g_std', 'b_std',
    'r_5_qtl', 'r_10_qtl', 'r_25_qtl', 'r_50_qtl', 'r_75_qtl', 'r_90_qtl', 'r_95_qtl',
    'g_5_qtl', 'g_10_qtl', 'g_25_qtl', 'g_50_qtl', 'g_75_qtl', 'g_90_qtl', 'g_95_qtl',
    'b_5_qtl', 'b_10_qtl', 'b_25_qtl', 'b_50_qtl', 'b_75_qtl', 'b_90_qtl', 'b_95_qtl',
    'r_g_correl', 'g_b_correl', 'b_r_correl',
    'Y', 'Z_prime', 'R_prime', 'Y_prime', 'X_prime', 'NDVI_c',
    'midday_r', 'midday_g', 'midday_b', 'midday_gcc', 'midday_rcc',
    'gcc_mean', 'gcc_std', 'gcc_50', 'gcc_75', 'gcc_90',
    'rcc_mean', 'rcc_std', 'rcc_50', 'rcc_75', 'rcc_90',
    'max_solar_elev', 'outlierflag_gcc_mean', 'outlierflag_gcc_50', 'outlierflag_gcc_75', 'outlierflag_gcc_90',
    'smooth_gcc_mean', 'smooth_gcc_50', 'smooth_gcc_75', 'smooth_gcc_90',
    'smooth_rcc_mean', 'smooth_rcc_50', 'smooth_rcc_75', 'smooth_rcc_90',
    'smooth_ci_gcc_mean', 'smooth_ci_gcc_50', 'smooth_ci_gcc_75', 'smooth_ci_gcc_90',
    'smooth_ci_rcc_mean', 'smooth_ci_rcc_50', 'smooth_ci_rcc_75', 'smooth_ci_rcc_90'
]
RESNET_TYPE = "resnet18"

@torch.no_grad()
def predict_future(target_date):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhenoCamLSTM(resnet=RESNET_TYPE, meta_dim=len(META_FEATURES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("date").reset_index(drop=True)

    last_days = df.tail(SEQUENCE_LENGTH)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    sample_imgs = []
    for idx in range(min(10, len(last_days))):
        filename = str(last_days.iloc[idx].get('image_filename') or last_days.iloc[idx].get('filename')).strip()
        img_path = IMAGE_FOLDER / filename
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            sample_imgs.append(img)
    avg_img = torch.stack(sample_imgs).mean(dim=0) if sample_imgs else torch.zeros((3, 224, 224))

    img_seq = []
    meta_seq = []

    for _, row in tqdm(last_days.iterrows(), total=len(last_days), desc="Preparing input sequence"):
        filename = str(row.get('image_filename') or row.get('filename')).strip()
        img_path = IMAGE_FOLDER / filename

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
        else:
            img = avg_img.clone()
        img_seq.append(img)

        meta_features = [float(row.get(col, 0.0)) if pd.notna(row.get(col)) else 0.0 for col in META_FEATURES]
        meta_seq.append(meta_features)

    img_seq = torch.stack(img_seq).unsqueeze(0).to(device)  # (1, T, C, H, W)
    meta_seq = torch.tensor(meta_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, meta_dim)

    output = model(img_seq, meta_seq)
    probs = F.softmax(output, dim=1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    print(f"\nPrediction for {target_date}:")
    print(f"- Predicted: {'Snow' if pred_label == 1 else 'No Snow'}")
    print(f"- Confidence: {confidence:.4f}")
    print(f"- Full Probabilities: {probs.squeeze().cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="Target future date to predict (YYYY-MM-DD)")
    args = parser.parse_args()

    predict_future(args.date)
