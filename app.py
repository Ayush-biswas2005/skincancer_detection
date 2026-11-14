
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "skin_model.pth"
UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=False
)
classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            error = "Please select an image."
        elif "." not in file.filename or (
            file.filename.rsplit(".", 1)[1].lower()
            not in ALLOWED_EXTENSIONS
        ):
            error = "Upload a JPG or PNG image."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
                image.thumbnail((1600, 1600))

                # Use a fixed server-generated filename.
                filename = "uploaded_image.jpg"
                saved_path = UPLOAD_DIR / filename
                image.save(saved_path, format="JPEG")
                image_url = "/static/uploads/" + filename

                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(tensor), dim=1
                    )[0]

                index = int(probabilities.argmax().item())
                prediction = {
                    "label": classes[index],
                    "score": float(probabilities[index]) * 100,
                }
            except Exception:
                error = "Could not process that image. Please try another."

    return render_template(
        "index.html",
        prediction=prediction,
        image_url=image_url,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
