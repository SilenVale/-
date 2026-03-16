
import torch
import sys
import os
import logging
import torch.nn as nn

# Adjust path to find core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MOCK CLIP ENCODER BEFORE IMPORTING ANYTHING ELSE
class MockClipEncoder(nn.Module):
    def __init__(self, model_name="mock"):
        super(MockClipEncoder, self).__init__()
        print("Initialized MockClipEncoder")
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # Return random features (batch, 50, 768)
        return torch.randn(batch_size, 50, 768).to(pixel_values.device)

# Patch the module in sys.modules so fusion_model imports our Mock
import core.models.clip_encoder
core.models.clip_encoder.ClipEncoder = MockClipEncoder

from core.models.fusion_model import MultimodalConflictModel
from core.config import *
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference():
    print("--- Starting Test Inference (with Mock CLIP) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    print("Initializing Model...")
    try:
        model = MultimodalConflictModel(
            bert_name=BERT_MODEL_NAME,
            clip_name="mock-clip", # Won't matter for mock
            freeze_bert_layers=FREEZE_BERT_LAYERS,
            num_classes=3
        ).to(device)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load weights manually to be sure
    # CHECKPOINT LOADING logic needs to handle mismatch keys for CLIP (since we mocked it)
    model_path = REAL_MODEL_PATH
    print(f"Loading weights from {model_path}")
    
    if not os.path.exists(model_path):
        print("Model path does not exist!")
        return
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded. Keys: {len(checkpoint)}")
        
        # We expect missing keys for CLIP since we mocked it
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys (total): {len(missing)}")
        
        # Verify FUSION keys are NOT missing
        missing_fusion = [k for k in missing if "fusion" in k]
        if missing_fusion:
             print(f"CRITICAL: Missing FUSION keys! {missing_fusion}")
        else:
             print("SUCCESS: Fusion keys loaded correctly.")

        # Check fusion weights in memory
        if hasattr(model, 'fusion') and hasattr(model.fusion, 'classifier'):
             w = model.fusion.classifier[0].weight
             print(f"Loaded Fusion Weight Stats: Mean={w.mean().item():.6f}, Std={w.std().item():.6f}")
             if w.std().item() == 0:
                 print("WARNING: Fusion weights have zero variance!")
             
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()

    # Create dummy inputs
    print("\n--- Running Dummy Inference ---")
    batch_size = 2
    seq_len = 20
    
    # Random text inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones((batch_size, seq_len)).to(device)
    
    # Random image inputs (mock encoder ignores content but shape matters for forward)
    # CLIP expects (batch, 3, 224, 224) usually
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print("Input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    
    # Hook to capture intermediate outputs
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook

    # Register hooks
    # model.text_encoder.register_forward_hook(get_activation('text_encoder'))
    # model.image_encoder is MockClipEncoder
    model.fusion.cross_attn.register_forward_hook(get_activation('fusion_attn'))
    model.fusion.classifier.register_forward_hook(get_activation('fusion_classifier'))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, pixel_values)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs # assuming dict or tensor

        print("\n--- Outputs ---")
        print(f"Logits:\n{logits}")
        
        probs = torch.softmax(logits, dim=1)
        print(f"Probs:\n{probs}")
        
        # Check intermediate activations
        print("\n--- Intermediate Activations ---")
        if 'fusion_attn' in activations:
            fa = activations['fusion_attn'][0] # (attn_output, weights) or tuple
            # Check if tuple
            if isinstance(activations['fusion_attn'], tuple):
                 fa = activations['fusion_attn'][0]
                 
            print(f"Fusion Attn Output: Shape={fa.shape}, Mean={fa.mean().item():.4f}, Std={fa.std().item():.4f}")

        if 'fusion_classifier' in activations:
            fc = activations['fusion_classifier']
            print(f"Classifier Output (Logits): Shape={fc.shape}, Mean={fc.mean().item():.4f}, Std={fc.std().item():.4f}")


if __name__ == "__main__":
    test_inference()
