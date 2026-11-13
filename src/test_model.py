import os
import torch

# Path to your weighted model
MODEL_PATH = os.path.join('model', 'skin_model_weighted.pth')

def test_load():
    if os.path.exists(MODEL_PATH):
        print(f"Success! Found model file at: {MODEL_PATH}")
        # Check file size
        file_size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Model file size: {file_size_kb:.2f} KB")
    else:
        print("Error: Model file not found. Check your folder path.")

if __name__ == '__main__':
    test_load()