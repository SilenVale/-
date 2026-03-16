# Multimodal Conflict Inspector (MCP Server)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch%20|%20MCP-orange.svg)

## 📖 项目介绍

**Multimodal Conflict Inspector** 是一个基于 **Model Context Protocol (MCP)** 架构的多模态智能体服务。它旨在检测图像与文本之间的情感和语义冲突（例如反讽、图文不符等），并提供基于**主动学习 (Active Learning)** 的样本筛选机制以及 **可解释性 (XAI)** 分析。

本项目不仅是一个图文分类模型，更是一个具备自我评估能力的智能辅助系统，包含以下核心特性：

*   **多模态融合检测**：利用 **BERT + CLIP** 双塔架构提取特征，通过 **Cross-Attention Fusion** 层深度融合，精准识别对齐、轻度冲突和严重冲突。
*   **MCP 协议集成**：完全符合 MCP 标准，提供 Tools、Resources 和 Prompts 接口，可轻松接入 Claude Desktop 或其他 AI 助手。
*   **主动学习闭环**：内置**不确定性采样 (Uncertainty Sampling)** 模块，自动筛选高熵值（最困惑）样本，辅助人工高效标注。
*   **可解释性分析**：提供基于注意力机制的热力图分析，解释模型判定冲突的依据。
*   **Gradio 可视化界面**：提供开箱即用的 Web UI，支持单样本检测和批量数据扫描。

---

## 🏗️ 项目结构

```text
YA_MCPServer_MMInspector_Submission/
├── client/                     # [客户端] 前端交互界面
│   └── gradio_app.py           # 基于 Gradio 的 Web UI，支持单样本检测与主动学习演示
│
├── core/                       # [核心层] 深度学习模型与业务逻辑
│   ├── checkpoints/            # 存放训练好的模型权重 (如 model_real.pth)
│   ├── data/                   # 数据集处理脚本与样例数据
│   ├── models/                 # 模型定义文件夹
│   │   ├── bert_classifier.py  # 文本编码器 (基于 BERT)
│   │   ├── clip_encoder.py     # 视觉编码器 (基于 CLIP-ViT)
│   │   └── fusion_model.py     # 多模态融合网络 (Cross-Attention)
│   ├── config.py               # 核心超参数配置 (模型路径、训练参数等)
│   ├── train.py                # 模型训练脚本
│   └── evaluate.py             # 模型评估与测试脚本
│
├── tools/                      # [工具层] MCP Tool 定义
│   ├── inspector_tools.py      # 核心工具集：封装冲突检测、XAI 解释、批量扫描等功能
│   └── hello_tool.py           # 示例工具
│
├── resources/                  # [资源层] MCP Resource 定义
│   └── dataset_resources.py    # 提供数据集统计信息 (dataset://stats) 等资源接口
│
├── prompts/                    # [提示词层] MCP Prompt 定义
│   └── analysis_prompts.py     # 定义分析图文冲突的 LLM 提示词模板
│
├── modules/                    # [模块层] 通用工具与配置模块
├── config.yaml                 # MCP Server 配置文件
├── env.yaml                    # 环境变量配置 (已脱敏)
├── server.py                   # MCP Server 启动入口
└── requirements.txt            # 项目依赖列表 (需自行生成)
```

---

## 🚀 快速开始

### 1. 环境准备

建议使用 Conda 创建独立的虚拟环境：

```bash
conda create -n mm_inspector python=3.10
conda activate mm_inspector
pip install -r requirements.txt
```

### 2. 启动 MCP Server

运行以下命令启动后端服务：

```bash
python server.py
```
服务启动后，支持通过 MCP 协议进行连接与调用。

### 3. 启动 Gradio 客户端

在新的终端窗口中运行：

```bash
python client/gradio_app.py
```
访问终端输出的本地链接（通常为 `http://127.0.0.1:7860`）即可使用可视化界面。

---

## 🛠️ 核心功能说明

### Tools (工具)
| 工具名称 | 功能描述 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| `inspect_multimodal_conflict` | **核心检测**：判断图文冲突等级 | `image_path`, `text` | JSON (Label, Confidence, Entropy) |
| `explain_decision` | **XAI 解释**：分析文本关键词权重 | `text`, `image_embedding` | Text Attention Weights |
| `batch_inspect` | **主动学习**：批量扫描并筛选高风险样本 | `batch_size` | List of Uncertain Samples |

### Resources (资源)
| URI | 描述 |
| :--- | :--- |
| `dataset://stats` | 获取当前数据集的统计分布信息 |
| `sample://{id}` | 获取特定样本的详细元数据 |

---

## 🧠 模型架构细节

本项目采用 **Late Fusion (晚期融合)** 变体架构：
1.  **Text Encoder**: 使用 `bert-base-chinese` 提取文本的上下文特征。
2.  **Image Encoder**: 使用 `openai/clip-vit-base-patch32` 提取图像的全局视觉特征。
3.  **Fusion Layer**: 自定义的 **Cross-Attention** 层，将文本作为 Query，图像特征作为 Key/Value，捕捉细粒度的语义对齐关系。

---

## ⚠️ 注意事项

*   **模型权重**: 本仓库默认不包含大型权重文件 (`.pth`)。请按照 `core/train.py` 自行训练，或联系作者获取演示权重。
*   **隐私说明**: 本项目代码已进行脱敏处理，`env.yaml` 中不包含任何真实的 API Key 或敏感配置。
*   **数据说明**: 示例数据仅用于演示格式，完整训练需使用 MintRec 或自定义数据集。

---

## 🤝 贡献与许可

欢迎提交 Issue 或 Pull Request 进行改进。本项目采用 MIT 许可证。

**开发者**: 
*   汪奕辰 (U202414702) - 全栈开发/模型架构
*   石俊贤 (U202414698) - 前端交互/数据清洗
