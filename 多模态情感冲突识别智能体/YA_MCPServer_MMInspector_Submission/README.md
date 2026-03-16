## YA_MCPServer_MMInspector

基于 MCP 协议的多模态冲突质检智能体，支持图文冲突检测、主动学习与 XAI 可解释性分析。


### Tool 列表

| 工具名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| inspect_multimodal_conflict | (核心) 基于训练好的 Fusion Model，判断图文冲突等级，输出置信度和熵值。 | image_path, text | JSON (Label, Confidence, Entropy) | 核心检测器 |
| explain_decision  | 利用 Attention 机制分析文本关键词权重，解释模型判定“冲突”的原因。 | text, image_embedding | Feature Importance Map | **XAI 可解释性** |
| batch_inspect | (主动学习) 批量扫描数据集样本，根据熵值自动筛选出高风险不确定样本。 | dataset_path, threshold | List[High_Risk_Samples] | **主动学习**模块 |
| suggest_annotation | 根据置信度分布与模型历史，为人工复核提供自然语言建议。 | confidence_score, label_dist | Recommendation String | 标注顾问 (Human-in-the-Loop) |


### Resource 列表

| 资源名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| dataset://stats | 实时读取 MintRec 数据集的标注分布与总量统计信息 | 无 (URI访问) | JSON (Total, Distribution) | 数据集概览 |
| sample://{sample_id} | 支持通过 ID 直接调取 MintRec 中的原始图文对进行深钻分析 | sample_id | Image + Text Metadata | 样本详情直达 |


### Prompts 列表

| 指令名称 | 功能描述 | 输入 | 输出 | 备注 |
| :------: | :------: | :--: | :--: | :--: |
| analyze_conflict | 生成多模态冲突分析的系统提示词，引导 LLM 结合心理学分析图文不符的深层含义。 | conflict_type, text, image_desc | Prompt Template | 辅助结果解释 |


### 项目结构

- core: 核心业务逻辑模块，包含：
    - models/: 定义了基于 **BERT-Base-Chinese** (文本) 和 **CLIP-ViT-Base** (图像) 的双塔融合神经网络模型 (FusionModel)。
    - checkpoints/: 包含 **真实模型** (model_real.pth, 用于常规检测) 和 **模拟模型** (用于演示主动学习)。
    - data/: 包含 **MintRec** 数据集转换脚本 (mintrec_real.jsonl, mintrec_200.jsonl)。
- 	ools: MCP 工具定义层，封装 inspector_tools.py 等模块，实现核心检测与 XAI 逻辑。
- client: **Gradio Web UI** 前端代码。
    - **Tab 1**: 单样本检测与 XAI 分析展示。
    - **Tab 2**: 主动学习 (Active Learning) 流程演示，支持虚拟构造样本的批量扫描。
- prompts: MCP 提示词定义，包含 analysis_prompts.py。
- 
esources: MCP 资源定义，包含 dataset_resources.py。
- config.yaml: 项目配置文件，配置了模型路径与服务端口。


### 其他需要说明的情况

- **Sops 密钥说明**: 本项目未使用任何外部付费 API Key。sops 模块仅作为框架保留，实际运行完全依赖本地训练的权重文件。
- **深度学习框架**: 使用了 **PyTorch** 以及 Hugging Face **Transformers** 库 (BERT, CLIP)。
- **机器学习模型**: 使用了 **多模态语义融合模型** (Fusion Model)。
    - **主动学习策略**: 引入了**不确定性采样 (Uncertainty Sampling)** 与 **熵值 (Entropy)** 计算，用于筛选高价值的“困惑”样本。
    - **数据策略**: 采用 **“真实基准 (MintRec Real) + 虚拟构造 (MintRec 200)”** 双轨制，验证模型在真实反讽场景与人工构造冲突场景下的表现。
