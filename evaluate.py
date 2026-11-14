
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
IMAGE_DIR = DATA_DIR / "images"
MODEL_PATH = ROOT / "model" / "skin_model.pth"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load the saved model checkpoint
checkpoint = torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=False
)
classes = checkpoint["classes"]
class_to_idx = {name: i for i, name in enumerate(classes)}

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


class SkinDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        missing_classes = set(self.df["dx"]) - set(class_to_idx)
        if missing_classes:
            raise ValueError(
                f"Unknown labels in test set: {missing_classes}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = IMAGE_DIR / f"{row['image_id']}.jpg"

        with Image.open(image_path) as img:
            image = transform(img.convert("RGB"))

        label = class_to_idx[row["dx"]]
        return image, label


test_dataset = SkinDataset(DATA_DIR / "test.csv")
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0
)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test macro-F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
))

# Save the report
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)

# Save a confusion matrix
cm = confusion_matrix(
    y_true, y_pred, labels=list(range(len(classes)))
)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues",
)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("HAM10000 Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("\nResults saved in:", RESULTS_DIR)
