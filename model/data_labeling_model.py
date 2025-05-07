import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter

# === Brightness Filter ===
def is_too_dark_avg_rgb(image_path, threshold=40):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    luminance = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    return luminance.mean() < threshold

# === Model Definition (2-head) ===
class MultiTaskResNet(torch.nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        base = models.resnet34(pretrained=True)
        self.features = torch.nn.Sequential(*list(base.children())[:-1])
        self.weather_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
        self.snow_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.weather_classifier(x), self.snow_classifier(x)

# === Dataset Class ===
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths, self.weather, self.snow = [], [], []
        self.transform = transform

        label_map = {
            "Sunny_No_Snow": (0, 0),
            "Sunny_With_Snow": (0, 1),
            "Cloudy_No_Snow": (1, 0),
            "Cloudy_With_Snow": (1, 1)
        }

        for folder, (w, s) in label_map.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for f in os.listdir(folder_path):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(folder_path, f)
                    if not is_too_dark_avg_rgb(full_path):
                        self.paths.append(full_path)
                        self.weather.append(w)
                        self.snow.append(s)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = datasets.folder.default_loader(str(self.paths[idx]))
        if self.transform:
            img = self.transform(img)
        return img, self.weather[idx], self.snow[idx]

# === Setup ===
data_path = "/home/nmichelotti/Desktop/data"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = WeatherDataset(data_path, transform)
train_len = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskResNet().to(device)

# === Class-Weighted Snow Loss ===
snow_counts = Counter(dataset.snow)
snow_tensor = torch.tensor([snow_counts[0], snow_counts[1]], dtype=torch.float32)
snow_weights = 1.0 / snow_tensor
snow_weights = snow_weights / snow_weights.sum()

criterion_weather = torch.nn.CrossEntropyLoss()
criterion_snow = torch.nn.CrossEntropyLoss(weight=snow_weights.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# === Training Loop ===
best_val_loss = float('inf')
for epoch in range(20):
    model.train()
    train_loss = 0
    for imgs, w_lbls, s_lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, w_lbls, s_lbls = imgs.to(device), w_lbls.to(device), s_lbls.to(device)
        optimizer.zero_grad()
        w_out, s_out = model(imgs)
        loss = criterion_weather(w_out, w_lbls) + criterion_snow(s_out, s_lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, w_lbls, s_lbls in val_loader:
            imgs, w_lbls, s_lbls = imgs.to(device), w_lbls.to(device), s_lbls.to(device)
            w_out, s_out = model(imgs)
            loss = criterion_weather(w_out, w_lbls) + criterion_snow(s_out, s_lbls)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/home/nmichelotti/Desktop/Senior Model/model/best_model.pth")
        print("Saved new best model.")

print("Training complete.")
