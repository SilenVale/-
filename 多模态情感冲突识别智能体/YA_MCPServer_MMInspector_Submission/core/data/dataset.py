import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, CLIPProcessor
from PIL import Image
import json
import os

class MultimodalDataset(Dataset):
    def __init__(self, data_file, image_dir, bert_name="bert-base-chinese", clip_name="openai/clip-vit-base-patch32", max_length=128):
        self.data = []
        self.image_dir = image_dir
        self.max_length = max_length
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                
        # 初始化 Tokenizer 和 Processor
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        
        # 标签映射 (对齐: 0, 轻度冲突: 1, 严重冲突: 2)
        self.label_map = {"aligned": 0, "mild": 1, "severe": 2}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        image_path = os.path.join(self.image_dir, item['image'])
        label_str = item.get('label', 'aligned') # 默认对齐
        label = self.label_map.get(label_str, 0)
        
        # 处理文本
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 处理图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # 如果图片不存在，创建一个全黑的占位图 (仅用于测试/演示)
            print(f"Warning: Image not found at {image_path}. Using dummy image.")
            image = Image.new('RGB', (224, 224), color='black')
            
        image_inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'id': item.get('id', str(idx))
        }

def get_dataloader(data_file, image_dir, batch_size=16, shuffle=True):
    dataset = MultimodalDataset(data_file, image_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
