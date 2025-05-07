import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- Darkness Detection ---
def is_too_dark_avg_rgb(image_path, threshold=40):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    luminance = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]
    avg_brightness = luminance.mean()
    return avg_brightness < threshold

# --- Model Definition (2 heads only) ---
class MultiTaskResNet(torch.nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        base = models.resnet34(pretrained=False)
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
        weather_out = self.weather_classifier(x)
        snow_out = self.snow_classifier(x)
        return weather_out, snow_out

# --- Device Selection ---
def get_best_device():
    if torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = get_best_device()

# --- Load model from state_dict ---
model_path = "/home/nmichelotti/Desktop/Senior Model/model/best_model.pth"
model = MultiTaskResNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Prediction Loop ---
folders = [
    "/home/nmichelotti/Desktop/Senior Model/Site Data/images_sagehen",
    "/home/nmichelotti/Desktop/Senior Model/Site Data/images_sagehen2",
    "/home/nmichelotti/Desktop/Senior Model/Site Data/images_sagehen3"
]

for folder in folders:
    results = []
    image_paths = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    
    for file in tqdm(image_paths, desc=f"Predicting {os.path.basename(folder)}"):
        try:
            full_path = os.path.join(folder, file)

            # Check for too dark before predicting
            if is_too_dark_avg_rgb(full_path):
                results.append({
                    "Image": file,
                    "WeatherPrediction": "Too Dark",
                    "WeatherConfidence": 0.0,
                    "SnowPrediction": "Too Dark",
                    "SnowConfidence": 0.0
                })
                continue

            image = Image.open(full_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                weather_out, snow_out = model(image_tensor)
                weather_probs = torch.softmax(weather_out, dim=1)[0]
                snow_probs = torch.softmax(snow_out, dim=1)[0]

                weather_idx = torch.argmax(weather_probs).item()
                snow_idx = torch.argmax(snow_probs).item()

                weather_label = ['Sunny', 'Cloudy'][weather_idx]
                snow_label = ['No Snow', 'Snow'][snow_idx]

                results.append({
                    "Image": file,
                    "WeatherPrediction": weather_label,
                    "WeatherConfidence": round(weather_probs[weather_idx].item() * 100, 2),
                    "SnowPrediction": snow_label,
                    "SnowConfidence": round(snow_probs[snow_idx].item() * 100, 2)
                })

        except Exception as e:
            print(f"[!] Error on {file}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(folder, "predictions.csv"), index=False)
    print(f"[✓] Saved predictions to: {folder}/predictions.csv")
