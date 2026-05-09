import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path("AMDNet23 Fundus Image Dataset for Age-Related Macular Degeneration Disease Detection/AMDNet23 Fundus Image Dataset for  Age-Related Macular Degeneration Disease Detection/AMDNet23 Dataset")

# Augmented transform for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
valid_dataset = datasets.ImageFolder(data_dir / "valid", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights="IMAGENET1K_V1")

# Add dropout before the final layer
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, 4)
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# ── Training ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_val_acc = 0
patience = 3          # stop if no improvement for 3 epochs
epochs_no_improve = 0

for epoch in range(20):  # allow up to 20 epochs
    # Train
    model.train()
    train_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc = 100 * correct / total

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1:02d} | Train acc: {train_acc:.1f}% | Val acc: {val_acc:.1f}%", end="")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "eye_disease_model.pth")
        print(" ✅ saved best model")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f" (no improvement {epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.1f}%")
            break

print("\nDone! Best model saved to eye_disease_model.pth")