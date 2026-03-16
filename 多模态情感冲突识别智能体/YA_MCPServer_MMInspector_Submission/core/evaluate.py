import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Adjust path to find core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from models.fusion_model import MultimodalConflictModel
from data.dataset import get_dataloader

def evaluate():
    print("--- Starting Evaluation ---")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing Model...")
    model = MultimodalConflictModel(
        bert_name=BERT_MODEL_NAME,
        clip_name=CLIP_MODEL_NAME,
        freeze_bert_layers=FREEZE_BERT_LAYERS,
        num_classes=3
    ).to(device)
    
    # Load weights
    model_path = REAL_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading weights from {model_path}...")
    try:
        # Compatibility loading
        try:
             checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
             checkpoint = torch.load(model_path, map_location=device)
             
        model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()

    # Load Test Data
    test_file = os.path.join(DATA_DIR, "mintrec_test.jsonl")
    if not os.path.exists(test_file):
        print(f"Error: Test data file not found at {test_file}")
        print("Please run 'python core/data/convert_data.py' first to generate it.")
        return
        
    print(f"Loading test data from {test_file}...")
    # Use smaller batch size for evaluation to be safe
    dataloader = get_dataloader(test_file, SAMPLES_DIR, batch_size=8, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask, pixel_values)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Total Accuracy: {accuracy:.4f}")
    
    target_names = ['Aligned (0)', 'Mild Conflict (1)', 'Severe Conflict (2)']
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\n(Row=True, Col=Predicted)")
    print("="*50)

if __name__ == "__main__":
    evaluate()
