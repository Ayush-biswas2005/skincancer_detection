
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "skin_model.pth"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(image_tensor), dim=1
        )[0]

    predicted_index = probabilities.argmax().item()

    return {
        "class": classes[predicted_index],
        "confidence": round(
            probabilities[predicted_index].item() * 100, 2
        ),
        "probabilities": {
            name: round(probabilities[i].item() * 100, 2)
            for i, name in enumerate(classes)
        }
    }


if __name__ == "__main__":
    image_path = input("Enter the path to a skin image: ").strip()
    result = predict_image(image_path)
    print("\nPrediction:", result["class"])
    print("Model confidence:", result["confidence"], "%")
    print("All class probabilities:", result["probabilities"])
