import torch
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", freeze_layers=10):
        super(BertClassifier, self).__init__()
        print(f"Loading BERT model: {model_name}...")
        try:
            # First try loading from cache/local
            self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
            print("BERT loaded from local cache.")
        except Exception:
            print("Local BERT not found, downloading...")
            self.bert = BertModel.from_pretrained(model_name)
            print("BERT downloaded.")
        
        # 分层微调：冻结前 N 层
        if freeze_layers > 0:
            for name, param in self.bert.named_parameters():
                if "encoder.layer" in name:
                    layer_num = int(name.split(".")[2])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                elif "embeddings" in name:
                    param.requires_grad = False
                    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 返回最后一层的隐藏状态 (batch_size, seq_len, hidden_size)
        return outputs.last_hidden_state
