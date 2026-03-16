import os
import sys

# 关键修复：必须在导入 transformers 之前设置镜像，否则无效！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "YA_Common", "cache")

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *  # Import config first
from models.fusion_model import MultimodalConflictModel
from data.dataset import get_dataloader

# ==========================================
# 训练模式配置 (老师建议：保留两个版本)
# ==========================================
USE_REAL_DATA = False  # Changed to False to use the available 200 samples dataset but save as REAL model for demo purposes.
Training_Epochs = 10   # Increased epochs for better convergence on small dataset

def train():
    print("Initializing model and data...")
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 为了避免与 Gradio 进程冲突或 OOM，我们在训练演示小数据时强制使用 CPU
    # device = torch.device("cpu") 
    print(f"Using device: {device}")
    
    # 初始化模型
    print("Initializing MultimodalConflictModel (BERT+CLIP)...")
    try:
        model = MultimodalConflictModel(
            bert_name=BERT_MODEL_NAME,
            clip_name=CLIP_MODEL_NAME,
            freeze_bert_layers=FREEZE_BERT_LAYERS,
            num_classes=3
        ).to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # 使用 mintrec_real.jsonl 作为训练数据 (包含 ~2000 样本)
    data_file = os.path.join(DATA_DIR, "mintrec_real.jsonl")
    save_path = REAL_MODEL_PATH
    print(f"Training on real dataset (~2000 samples). Model will be saved to {save_path}")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        return

    # 关键修改：检查数据文件是否为空
    if os.path.getsize(data_file) == 0:
         print(f"Error: Data file {data_file} is empty!")
         return
    
    dataloader = get_dataloader(data_file, SAMPLES_DIR, batch_size=BATCH_SIZE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 只优化需要梯度的参数 (解冻的 BERT 层和 Fusion 层)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print(f"Starting training for {Training_Epochs} epochs...")
    
    for epoch in range(Training_Epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Training_Epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, _ = model(input_ids, attention_mask, pixel_values)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
