import torch
import torch.nn as nn
from .bert_classifier import BertClassifier
from .clip_encoder import ClipEncoder

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, num_heads=8, num_classes=3):
        super(CrossAttentionFusion, self).__init__()
        
        # 文本作为 Query，图像作为 Key 和 Value
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)
        
        # 维度对齐（如果 CLIP 和 BERT 维度不同，需要线性层对齐）
        self.image_proj = nn.Linear(image_dim, text_dim) if image_dim != text_dim else nn.Identity()
        
        # 分类头 (三级分类：对齐/轻度/严重)
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_features, image_features, text_mask=None):
        """
        text_features: (batch_size, text_seq_len, text_dim)
        image_features: (batch_size, image_seq_len, image_dim)
        text_mask: (batch_size, text_seq_len) - 1 for valid, 0 for padding
        """
        # 投影图像特征以匹配文本维度
        image_features = self.image_proj(image_features)
        
        # Cross-Attention: Text (Q), Image (K, V)
        # attn_output: (batch_size, text_seq_len, text_dim)
        # attn_weights: (batch_size, text_seq_len, image_seq_len) - 用于可解释性
        attn_output, attn_weights = self.cross_attn(
            query=text_features,
            key=image_features,
            value=image_features,
            key_padding_mask=None # 假设图像没有 padding
        )
        
        # 聚合文本序列特征 (这里简单取 CLS token，即第一个 token)
        # 也可以使用 mean pooling
        cls_output = attn_output[:, 0, :]
        
        # 分类预测
        logits = self.classifier(cls_output)
        
        return logits, attn_weights

class MultimodalConflictModel(nn.Module):
    def __init__(self, bert_name="bert-base-chinese", clip_name="openai/clip-vit-base-patch32", freeze_bert_layers=10, num_classes=3):
        super(MultimodalConflictModel, self).__init__()
        self.text_encoder = BertClassifier(model_name=bert_name, freeze_layers=freeze_bert_layers)
        self.image_encoder = ClipEncoder(model_name=clip_name)
        
        # BERT base 和 CLIP ViT-B/32 的隐藏层维度都是 768
        self.fusion = CrossAttentionFusion(text_dim=768, image_dim=768, num_classes=num_classes)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        # 提取文本特征
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # 提取图像特征 (冻结，不需要梯度)
        with torch.no_grad():
            image_features = self.image_encoder(pixel_values)
            
        # 跨模态融合与分类
        logits, attn_weights = self.fusion(text_features, image_features, text_mask=attention_mask)
        
        return logits, attn_weights
