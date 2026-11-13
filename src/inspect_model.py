import os
import torch

MODEL_PATH = os.path.join('model', 'skin_model_weighted.pth')

def inspect_weights():
    print("Inspecting model state dictionary...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # If the file saved the entire dict or just state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and not any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        # Might be a training checkpoint dictionary
        print("Keys available in checkpoint dictionary:", list(checkpoint.keys()))
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint

    print(f"\nTotal layers/parameters found: {len(state_dict)}")
    print("\nFirst 10 layer keys in the model:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")

if __name__ == '__main__':
    inspect_weights()