from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "images"
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Training settings
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# Dataset
class SkinDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        required = {"image_id", "dx"}
        if not required.issubset(self.df.columns):
            raise ValueError(
                f"CSV must contain {required}"
            )

        self.df = self.df.dropna(subset=["image_id", "dx"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = DATA_DIR / f"{row['image_id']}.jpg"

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = self.transform(image)

        label = class_to_idx[row["dx"]]

        return image, label


# Load existing splits
train_df = pd.read_csv(DATASET_DIR / "train.csv")
val_df = pd.read_csv(DATASET_DIR / "validation.csv")

classes = sorted(
    pd.read_csv(DATASET_DIR / "HAM10000_metadata.csv")["dx"]
    .dropna()
    .unique()
)

class_to_idx = {
    name: i for i, name in enumerate(classes)
}

print("Classes:", classes)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# Create datasets and loaders
train_dataset = SkinDataset(
    DATASET_DIR / "train.csv",
    train_transform
)

val_dataset = SkinDataset(
    DATASET_DIR / "validation.csv",
    eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Load pretrained model
weights = MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Freeze feature extractor for the first version
for parameter in model.features.parameters():
    parameter.requires_grad = False

input_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    input_features,
    len(classes)
)

model = model.to(DEVICE)

# Balance classes using training labels only
counts = train_df["dx"].value_counts()

class_weights = torch.tensor(
    [
        len(train_df) / (
            len(classes) * max(counts.get(c, 0), 1)
        )
        for c in classes
    ],
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# Training loop
best_f1 = -1.0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    train_true = []
    train_pred = []

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        train_true.extend(labels.cpu().tolist())
        train_pred.extend(
            outputs.argmax(dim=1).detach().cpu().tolist()
        )

    model.eval()
    val_true = []
    val_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))

            val_true.extend(labels.tolist())
            val_pred.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )

    train_loss = total_loss / len(train_dataset)
    train_acc = accuracy_score(train_true, train_pred)
    val_acc = accuracy_score(val_true, val_pred)

    val_f1 = f1_score(
        val_true,
        val_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.4f} | "
        f"Val accuracy: {val_acc:.4f} | "
        f"Val macro-F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "architecture": "mobilenet_v3_small",
            "image_size": 224
        }, MODEL_DIR / "skin_model.pth")

print("\nTraining finished.")
print("Model saved to:", MODEL_DIR / "skin_model.pth")