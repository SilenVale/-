
import torch
import sys
import os
import numpy as np

# Adjust path to find core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *

def check_checkpoint_weights():
    # Force use of local path
    real_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "checkpoints", "model_real.pth")
    print(f"Checking model path: {real_model_path}")
    
    if not os.path.exists(real_model_path):
        print(f"Error: Model file not found at {real_model_path}")
        return

    print(f"Loading checkpoint...")
    try:
        checkpoint = torch.load(real_model_path, map_location="cpu")
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("\n--- Model Weights Analysis ---")
    keys = list(checkpoint.keys())
    print(f"Total keys in checkpoint: {len(keys)}")
    
    # Check for keys that contain 'fusion'
    fusion_keys = [k for k in keys if "fusion" in k]
    print(f"Fusion keys found: {len(fusion_keys)}")
    
    if not fusion_keys:
        print("WARNING: No explicit 'fusion' keys found in checkpoint.")
        return

    fusion_weights = []
    
    for key in fusion_keys:
        value = checkpoint[key]
        if "weight" in key and value.dim() > 1:
            mean_val = value.float().mean().item()
            std_val = value.float().std().item()
            min_val = value.min().item()
            max_val = value.max().item()
            
            print(f"\nLayer: {key}")
            print(f"  Shape: {value.shape}")
            print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
            
            if std_val == 0:
                print(f"  WARNING: Layer {key} has zero variance (constant values)!")
            
            fusion_weights.append(value)
        elif "classifier" in key:
             print(f"\nLayer: {key}")
             print(f"  Shape: {value.shape}")
             print(f"  Values: {value}")


if __name__ == "__main__":
    check_checkpoint_weights()

