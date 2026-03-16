import os

# 设置 HuggingFace 国内镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 缓存路径
# CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "YA_Common", "cache")
# os.environ["HF_HOME"] = CACHE_DIR

# 项目根目录 (指向 YA_MCPServer_MMInspector)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 核心逻辑目录 (指向 YA_MCPServer_MMInspector/core)
CORE_DIR = os.path.join(BASE_DIR, "core")

# 数据与模型路径
DATA_DIR = os.path.join(CORE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
CHECKPOINT_DIR = os.path.join(CORE_DIR, "checkpoints")

# 确保目录存在
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 模型配置
BERT_MODEL_NAME = "bert-base-chinese"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# 训练超参数 (100分版要求)
FREEZE_BERT_LAYERS = 10  # 冻结前10层，解冻后2层
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3

# 模型保存路径 (老师建议：保留两个版本)
MOCK_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_mock.pth")
REAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_real.pth")

# ==========================================
# 演示模式配置 (用于快速切换 Gradio 界面使用的模型)
# ==========================================
# 设置为 True: 强制使用 model_real.pth (展示高准确率)
# 设置为 False: 强制使用 model_mock.pth (演示主动学习 UI 流程)
USE_REAL_MODEL_FOR_DEMO = True

# 主动学习配置
UNCERTAINTY_THRESHOLD = 0.6 # 熵值高于此阈值认为是不确定样本
