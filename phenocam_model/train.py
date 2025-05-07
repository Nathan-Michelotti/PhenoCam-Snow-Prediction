import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from model import PhenoCamLSTM
from dataset import PhenoCamSequenceDataset

# Editable parameters
DATA_ROOT = Path("../Site Data")
MODEL_OUTPUT = "PhenoCam_LSTM_model.pth"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
SEQUENCE_LENGTH = 90
PREDICT_AHEAD_DAYS = 14
RESNET_TYPE = "resnet18"

META_FEATURES = [
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

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset_list = []
    for entry in [{
        "image_dir": DATA_ROOT / "images_sagehen",
        "metadata_csvs": [
            Path("../data_orginization/sagehen/3_sagehen_EN_1000_roistats.csv"),
            Path("../data_orginization/sagehen/4_sagehen_EN_1000_3day.csv"),
            Path("../data_orginization/sagehen/6_sagehen_EN_1000_ndvi_roistats_4_filtered.csv")
        ]
    }]:
        ds = PhenoCamSequenceDataset(
            image_dir=entry["image_dir"],
            metadata_csvs=entry["metadata_csvs"],
            meta_features=META_FEATURES,
            sequence_length=SEQUENCE_LENGTH,
            predict_ahead_days=PREDICT_AHEAD_DAYS
        )
        dataset_list.append(ds)

    if len(dataset_list) == 1:
        dataset = dataset_list[0]
    else:
        dataset = ConcatDataset(dataset_list)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PhenoCamLSTM(
        resnet=RESNET_TYPE,
        meta_dim=len(META_FEATURES),
        hidden_dim=128,
        n_classes=2,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        callbacks=[TQDMProgressBar(refresh_rate=1)]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")
