import os
import json
from . import YA_MCPServer_Resource
from core.config import *
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@YA_MCPServer_Resource(uri="dataset://stats", name="dataset_stats", description="Returns dataset statistics.")
def get_dataset_stats() -> str:
    """
    返回 MintRec 数据集的统计信息（各类别样本数）。
    
    Returns:
        str: 包含数据集统计信息 (Distribution) 的 JSON 字符串。
    """
    try:
        data_path = os.path.join(DATA_DIR, "mintrec_200.jsonl")
        if not os.path.exists(data_path):
            return json.dumps({"error": f"Dataset file not found at {data_path}"})
            
        stats = {
            "total_samples": 0,
            "labels": {"0": 0, "1": 0, "2": 0},
            "missing_images": 0
        }
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    stats["total_samples"] += 1
                    label = str(data.get("label", "unknown"))
                    if label in stats["labels"]:
                        stats["labels"][label] += 1
                    
                    img_path = data.get("image", "")
                    # 如果不是绝对路径，拼接 SAMPLES_DIR
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(SAMPLES_DIR, img_path)
                    
                    if not os.path.exists(img_path):
                         stats["missing_images"] += 1
                except json.JSONDecodeError:
                    continue
                    
        result = {
            "dataset_name": "MintRec (Multimodal Intent Recognition)",
            "description": "A dataset for detecting intent and conflict in multimodal dialogue.",
            "statistics": stats
        }
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in get_dataset_stats: {str(e)}")
        return json.dumps({"error": str(e)})

@YA_MCPServer_Resource(
    uri="sample://{sample_id}",
    name="get_sample_by_id",
    description="Get detailed sample info by ID (text, image, labels)."
)
def get_sample_by_id(sample_id: str) -> str:
    """
    根据 ID 返回特定样本的详细信息（文本、图像路径、标签）。
    
    Args:
        sample_id (str): 样本的唯一标识符。
        
    Returns:
        str: 包含样本详细信息的 JSON 字符串。
    """
    try:
        data_path = os.path.join(DATA_DIR, "mintrec_200.jsonl")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if str(data.get("id")) == str(sample_id):
                        return json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    continue
        
        return json.dumps({"error": f"Sample with ID {sample_id} not found."})
        
    except Exception as e:
        logger.error(f"Error in get_sample_by_id: {str(e)}")
        return json.dumps({"error": str(e)})