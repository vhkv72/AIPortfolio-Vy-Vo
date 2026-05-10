import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ── Load model ────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, 4)
)
model.load_state_dict(torch.load("eye_disease_model.pth", map_location="cpu"))
model.eval()

classes = ["amd", "cataract", "diabetes", "normal"]

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path("AMDNet23 Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection/AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection/AMDNet23 Dataset")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_dataset = datasets.ImageFolder(data_dir / "valid", transform=val_transform)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# ── Run predictions ───────────────────────────────────────────────────────────
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in valid_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# ── Build confusion matrix ────────────────────────────────────────────────────
matrix = np.zeros((4, 4), dtype=int)
for true, pred in zip(all_labels, all_preds):
    matrix[true][pred] += 1

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap="Blues")

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)
ax.set_xlabel("Predicted", fontsize=13)
ax.set_ylabel("Actual", fontsize=13)
ax.set_title("Confusion Matrix — Validation Set", fontsize=14)

# Add numbers inside each cell
for i in range(4):
    for j in range(4):
        color = "white" if matrix[i][j] > matrix.max() / 2 else "black"
        ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                fontsize=14, color=color, fontweight="bold")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved to confusion_matrix.png")

# ── Print per-class accuracy ──────────────────────────────────────────────────
print("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    correct = matrix[i][i]
    total = matrix[i].sum()
    print(f"  {cls:<10} {correct}/{total}  ({100*correct/total:.1f}%)")