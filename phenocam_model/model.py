import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class PhenoCamLSTM(pl.LightningModule):
    def __init__(self, resnet, meta_dim, hidden_dim=128, n_classes=2, lr=1e-4, weight_decay=0.01):
        super().__init__()
        self.save_hyperparameters()

        backbone = getattr(models, resnet)(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.image_feature_dim = backbone.fc.in_features
        self.input_dim = self.image_feature_dim + meta_dim

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.metric = MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x_img_seq, x_meta_seq):
        B, T, C, H, W = x_img_seq.shape
        img_seq_flat = x_img_seq.view(B*T, C, H, W)
        img_feats = self.feature_extractor(img_seq_flat).flatten(1)
        img_feats = img_feats.view(B, T, -1)

        x = torch.cat([img_feats, x_meta_seq], dim=2)
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        return self.classifier(final_out)

    def training_step(self, batch, batch_idx):
        x_img_seq, x_meta_seq, y = batch
        yhat = self(x_img_seq, x_meta_seq)
        loss = F.cross_entropy(yhat, y)
        acc = self.metric(yhat.argmax(dim=1), y)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_img_seq, x_meta_seq, y = batch
        yhat = self(x_img_seq, x_meta_seq)
        loss = F.cross_entropy(yhat, y)
        acc = self.metric(yhat.argmax(dim=1), y)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
