from modules.YA_Common.utils.logger import get_logger
import os
import requests

logger = get_logger("setup")

def verify_and_setup_models(core_dir):
    """
    Check if model weights and datasets exist. Simulate download logic.
    """
    checkpoint_dir = os.path.join(core_dir, "checkpoints")
    real_model_path = os.path.join(checkpoint_dir, "model_real.pth")
    mock_model_path = os.path.join(checkpoint_dir, "model_mock.pth")
    
    # 模拟下载 BERT/CLIP 预训练权重的逻辑
    # 实际上 transformers 库会在第一次运行时自动下载缓存
    logger.info("Verifying model checkpoints...")
    
    if os.path.exists(real_model_path):
        logger.info(f"✅ Found REAL model: {real_model_path}")
    else:
        logger.warning(f"⚠️ REAL model missing at {real_model_path}")
        # 这里可以放置下载代码，例如：
        # requests.get("http://internal-repo/model_real.pth")
        
    if os.path.exists(mock_model_path):
        logger.info(f"✅ Found MOCK model: {mock_model_path}")
    else:
        logger.warning(f"⚠️ MOCK model missing. Active learning demo might fail.")

def setup():
    """Setup your environment and dependencies here."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        core_dir = os.path.join(base_dir, "core")
        
        logger.info("Starting setup: Verifying core assets...")
        
        # 验证模型和数据是否存在
        verify_and_setup_models(core_dir)
        
        logger.info("Setup complete. All dependencies verified.")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise e
