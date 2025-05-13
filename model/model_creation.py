import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define MultiTask Model
class MultiTaskResNet(torch.nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        self.base_model = models.resnet34(pretrained=True)

        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        self.features = torch.nn.Sequential(*list(self.base_model.children())[:-1])

        self.weather_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 2)
        )
        self.snow_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.weather_classifier(x), self.snow_classifier(x)

# Custom dataset including "Too Dark" images as unknown (0, 0)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.weather_labels = []
        self.snow_labels = []
        self.transform = transform

        category_map = {
            "Sunny_With_Snow": (0, 1),
            "Sunny_No_Snow": (0, 0),
            "Cloudy_With_Snow": (1, 1),
            "Cloudy_No_Snow": (1, 0)
        }

        for category, (w_label, s_label) in category_map.items():
            full_path = os.path.join(root_dir, category)
            if os.path.exists(full_path):
                for fname in os.listdir(full_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(full_path, fname))
                        self.weather_labels.append(w_label)
                        self.snow_labels.append(s_label)

        # Include 'Too Dark' folder with label (0, 0) representing unknown
        too_dark_path = os.path.join(root_dir, "to_dark")
        if os.path.exists(too_dark_path):
            for fname in os.listdir(too_dark_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(too_dark_path, fname))
                    self.weather_labels.append(0)
                    self.snow_labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = datasets.folder.default_loader(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.weather_labels[idx], self.snow_labels[idx]

# Set paths
dataset_path = "/home/nmichelotti/Desktop/data"

# Image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomDataset(dataset_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskResNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

# Training loop
best_val_loss = float('inf')
for epoch in range(30):
    model.train()
    running_loss = 0
    correct_weather = 0
    correct_snow = 0
    total = 0
    for images, weather_labels, snow_labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images = images.to(device)
        weather_labels = weather_labels.to(device)
        snow_labels = snow_labels.to(device)

        optimizer.zero_grad()
        weather_out, snow_out = model(images)
        loss = criterion(weather_out, weather_labels) + criterion(snow_out, snow_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        correct_weather += (torch.argmax(weather_out, 1) == weather_labels).sum().item()
        correct_snow += (torch.argmax(snow_out, 1) == snow_labels).sum().item()
        total += weather_labels.size(0)

    train_loss = running_loss / len(train_loader)
    weather_acc = correct_weather / total * 100
    snow_acc = correct_snow / total * 100

    model.eval()
    val_loss = 0
    val_correct_weather = 0
    val_correct_snow = 0
    val_total = 0
    with torch.no_grad():
        for images, weather_labels, snow_labels in val_loader:
            images = images.to(device)
            weather_labels = weather_labels.to(device)
            snow_labels = snow_labels.to(device)

            weather_out, snow_out = model(images)
            loss = criterion(weather_out, weather_labels) + criterion(snow_out, snow_labels)
            val_loss += loss.item()
            val_correct_weather += (torch.argmax(weather_out, 1) == weather_labels).sum().item()
            val_correct_snow += (torch.argmax(snow_out, 1) == snow_labels).sum().item()
            val_total += weather_labels.size(0)

    val_loss /= len(val_loader)
    val_weather_acc = val_correct_weather / val_total * 100
    val_snow_acc = val_correct_snow / val_total * 100

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Weather Acc = {weather_acc:.2f}%, Snow Acc = {snow_acc:.2f}%")
    print(f"            Val Loss = {val_loss:.4f}, Weather Acc = {val_weather_acc:.2f}%, Snow Acc = {val_snow_acc:.2f}%")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model.")

print("Training complete.")
