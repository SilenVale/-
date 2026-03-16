
import torch
import sys
import os

# Adjust path to find core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import *

def inspect_checkpoint():
    print(f"Checking checkpoint file: {REAL_MODEL_PATH}")
    if not os.path.exists(REAL_MODEL_PATH):
        print("File not found.")
        return

    try:
        # Load state dict ONLY
        state_dict = torch.load(REAL_MODEL_PATH, map_location="cpu")
        keys = list(state_dict.keys())
        print(f"Total keys in checkpoint: {len(keys)}")
        
        # Check for expected prefixes
        text_keys = [k for k in keys if k.startswith("text_encoder")]
        image_keys = [k for k in keys if k.startswith("image_encoder")]
        fusion_keys = [k for k in keys if k.startswith("fusion")]
        
        print(f"Text Encoder keys: {len(text_keys)}")
        print(f"Image Encoder keys: {len(image_keys)}")
        print(f"Fusion keys: {len(fusion_keys)}")

        if len(text_keys) == 0:
             print("WARNING: 'text_encoder' prefix missing. The model expected 'text_encoder.bert...' keys.")
             print(f"Found keys starting with 'bert': {len([k for k in keys if k.startswith('bert')])}")
        
        if len(image_keys) == 0:
             print("WARNING: 'image_encoder' prefix missing. The model expected 'image_encoder.clip...' keys.")
             print(f"Found keys starting with 'clip': {len([k for k in keys if k.startswith('clip')])}")

        if len(fusion_keys) == 0:
             print("WARNING: 'fusion' prefix missing.")
             print(f"Found keys starting with 'classifier': {len([k for k in keys if k.startswith('classifier')])}")
             
        # Check for fusion related keys
        fusion_keys = [k for k in keys if "fusion" in k]
        print(f"Keys containing 'fusion': {len(fusion_keys)}")
        for k in fusion_keys:
            print(f" - {k}")
            
        # Check for classifier related keys if fusion is missing
        classifier_keys = [k for k in keys if "classifier" in k]
        print(f"Keys containing 'classifier' (outside fusion?): {len(classifier_keys)}")
        for k in classifier_keys[:10]:
             print(f" - {k}")

        if len(fusion_keys) == 0:
            print("\nCRITICAL: No 'fusion' keys found!")
            print("This suggests the checkpoint saved only the BERT/CLIP backbones (or parts of them) and not the training head.")
            print("Did you save state_dict() on the whole model or just a sub-module?")
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
