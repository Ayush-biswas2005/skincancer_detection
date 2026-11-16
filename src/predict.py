"""
Skin disease inference module with multi-stage image validation.
Uses MobileNetV3-Small trained on HAM10000 dataset for dermatological classification,
guarded by an upstream multi-stage validation pipeline.
"""

import os
import torch
from torchvision import models, transforms
from PIL import Image

from src.skin_validator import ValidationPipeline, ValidationResult

MODEL_PATH = os.path.join('model', 'skin_model_weighted.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SkinDiseasePredictor:
    """Predicts dermatological conditions with strict upstream validation."""

    def __init__(self, model_path=MODEL_PATH):
        self.device = device
        self.model = None
        self.classes = []
        self.image_size = 224
        self.transform = None

        # Upstream Multi-Stage Validation Pipeline
        self.validator = ValidationPipeline(self.device)

        self._load_model_and_metadata(model_path)

    def _load_model_and_metadata(self, path):
        print(f"Loading diagnostic model checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)

        self.classes = checkpoint.get('classes', [
            'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
        ])
        self.image_size = checkpoint.get('image_size', 224)
        architecture = checkpoint.get('architecture', 'mobilenet_v3_small')

        print(f"Detected Architecture: {architecture}")
        print(f"Detected Image Size: {self.image_size}x{self.image_size}")
        print(f"Detected Diagnostic Classes: {self.classes}")

        if 'mobilenet_v3_small' in architecture.lower():
            self.model = models.mobilenet_v3_small(weights=None)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, len(self.classes))
        elif 'mobilenet_v2' in architecture.lower():
            self.model = models.mobilenet_v2(weights=None)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, len(self.classes))
        else:
            self.model = models.resnet18(weights=None)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, len(self.classes))

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Diagnostic model successfully loaded into memory.")

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path: str):
        """
        Executes the full validated prediction pipeline:
          Image Validation -> Human Skin Detection -> Quality Check -> Disease Classification
        
        Returns:
            tuple: (is_valid: bool, prediction: str or None, confidence: str or None, message: str)
        """
        # Step 1: Upstream multi-stage validation gate
        val_result: ValidationResult = self.validator.validate(image_path)
        if not val_result.is_valid:
            print(f"[Predictor] Image rejected at stage [{val_result.stage}]: {val_result.message}")
            return False, None, None, val_result.message

        # Step 2: Diagnostic Model Inference (only executed for validated skin specimens)
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, predicted_idx = torch.max(probabilities, 0)

            confidence_value = conf.item() * 100.0
            predicted_class = (
                self.classes[predicted_idx.item()]
                if self.classes else str(predicted_idx.item())
            )
            confidence_str = f"{confidence_value:.2f}%"

            return True, predicted_class, confidence_str, "Analysis completed successfully."

        except Exception as e:
            print(f"[Predictor] Inference runtime error: {e}")
            return False, None, None, "An unexpected error occurred during diagnostic analysis."


# Global instance for app usage
predictor = SkinDiseasePredictor()
