import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ── Load model ────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("eye_disease_model.pth", map_location="cpu"))
model.eval()

classes = ["amd", "cataract", "diabetes", "normal"]

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {classes[pred].upper()}")
    print("\nConfidence scores:")
    for cls, prob in zip(classes, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<10} {prob*100:5.1f}%  {bar}")

# ── Run ───────────────────────────────────────────────────────────────────────
data_dir = Path("AMDNet23 Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection/AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection/AMDNet23 Dataset")

# Pick one image from each class in the validation set
for cls in ["amd", "cataract", "diabetes", "normal"]:
    folder = data_dir / "valid" / cls
    sample = list(folder.iterdir())[0]  # first image in the folder
    predict(sample)