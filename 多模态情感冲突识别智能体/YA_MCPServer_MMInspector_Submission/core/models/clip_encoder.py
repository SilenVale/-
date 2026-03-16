import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ClipEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(ClipEncoder, self).__init__()
        print(f"Loading CLIP model: {model_name}...")
        try:
            # First try loading from cache/local with safetensors (default)
            self.clip = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True, local_files_only=True)
            print("CLIP loaded from local cache (safetensors).")
        except Exception as e1:
            print(f"Local safetensors failed: {e1}. Trying bin...")
            try:
                self.clip = CLIPVisionModel.from_pretrained(model_name, use_safetensors=False, local_files_only=True)
                print("CLIP loaded from local cache (bin).")
            except Exception as e2:
                print(f"Local CLIP not found, downloading... (Error: {e2})")
                self.clip = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
                print("CLIP downloaded.")
        
        # 冻结 CLIP 模型的所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        outputs = self.clip(pixel_values=pixel_values)
        # 返回最后一层的隐藏状态 (batch_size, seq_len, hidden_size)
        # 对于 ViT，seq_len = 1 (CLS token) + num_patches
        return outputs.last_hidden_state
