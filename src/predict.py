import os
import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = os.path.join('model', 'skin_model_weighted.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkinDiseasePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = device
        self.model = None
        self.classes = []
        self.image_size = 224
        self.transform = None
        self._load_model_and_metadata(model_path)

    def _load_model_and_metadata(self, path):
        print(f"Loading model checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Extract metadata saved by Member 2
        self.classes = checkpoint.get('classes', [])
        self.image_size = checkpoint.get('image_size', 224)
        architecture = checkpoint.get('architecture', 'mobilenet_v3_small')
        
        print(f"Detected Architecture: {architecture}")
        print(f"Detected Image Size: {self.image_size}x{self.image_size}")
        print(f"Detected Classes: {self.classes}")

        # Instantiate MobileNet V3 Small matching Member 2's custom configuration
        if 'mobilenet_v3_small' in architecture.lower():
            self.model = models.mobilenet_v3_small(weights=None)
            # Adjust final classifier layer to match the number of classes
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

        # Load the weights state dictionary safely
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully into memory!")

        # Setup dynamic preprocessing transforms matching training specs
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = self.classes[predicted_idx.item()] if self.classes else str(predicted_idx.item())
            confidence = f"{conf.item() * 100:.2f}%"
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Inference error: {e}")
            return "Error during inference", "0%"

# Global instance for app usage
predictor = SkinDiseasePredictor()