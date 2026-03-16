import gradio as gr
import os
import sys

# 【强制】设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import json
import shutil
import logging

# 获取项目根目录 (YA_MCPServer_MMInspector)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# 导入核心模块
from core.config import *
from tools.inspector_tools import detect_conflict, explain_decision, batch_inspect, suggest_annotation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保 samples 目录存在
os.makedirs(SAMPLES_DIR, exist_ok=True)

def process_single_sample(text, image):
    """处理单个样本的预测和解释"""
    if not text or image is None:
        return "Please provide both text and image.", "", "", ""
        
    # 保存上传的图片到 samples 目录
    image_filename = os.path.basename(image)
    target_path = os.path.join(SAMPLES_DIR, image_filename)
    
    # 如果图片不在 samples 目录中，复制过去
    if not os.path.exists(target_path) or os.path.abspath(image) != os.path.abspath(target_path):
        try:
            shutil.copy2(image, target_path)
        except shutil.SameFileError:
            pass
            
    # 1. 调用 detect_conflict tool (返回 JSON 字符串)
    detect_json = detect_conflict(text, target_path)
    detect_result = json.loads(detect_json)
    
    if "error" in detect_result:
        return f"Error: {detect_result['error']}", "", "", ""
        
    prediction = detect_result["prediction"]
    confidence = detect_result["confidence"]
    entropy = detect_result["entropy"]
    
    # 格式化预测结果
    result_md = f"### 🎯 Prediction: **{prediction}**\n"
    result_md += f"- **Confidence**: {confidence:.2%}\n"
    result_md += f"- **Uncertainty (Entropy)**: {entropy:.4f}\n"
    
    if detect_result["needs_human_review"]:
        result_md += "\n⚠️ **High Uncertainty Detected! Human review recommended.**"
        
    # 2. 调用 explain_decision tool (返回 JSON 字符串)
    explain_json = explain_decision(text, target_path)
    explain_result = json.loads(explain_json)
    
    explanation_md = "### 🔍 Attention Analysis\n"
    if "error" not in explain_result and "top_words" in explain_result:
        explanation_md += "Words with highest attention weights:\n"
        for item in explain_result["top_words"]:
            # 使用 HTML 渲染一个简单的热力图条
            weight = item['weight']
            # 根据权重调整红色深浅 (rgba)
            color = f"rgba(255, 99, 71, {weight})" 
            explanation_md += f"- <span style='background-color: {color}; padding: 2px 5px; border-radius: 3px;'>**{item['word']}**</span> (Weight: {weight:.2f})\n"
    else:
        explanation_md += "Explanation not available."
        
    # 3. 调用 suggest_annotation tool (直接返回字符串)
    suggestion = suggest_annotation(prediction, confidence, entropy)
    suggestion_md = f"### 💡 AI Advisor\n{suggestion}"
    
    # 原始 JSON 输出 (展示 MCP 协议)
    raw_json = json.dumps({
        "detection": detect_result,
        "explanation": explain_result
    }, indent=2, ensure_ascii=False)
    
    return result_md, explanation_md, suggestion_md, raw_json

def run_active_learning(batch_size):
    """运行主动学习批量扫描"""
    # 调用 batch_inspect tool (返回 JSON 字符串)
    result_json = batch_inspect(int(batch_size))
    result = json.loads(result_json)
    
    if "error" in result:
        return f"Error: {result['error']}", None, "" 
        
    output_md = f"### 📊 Active Learning Scan Results\n"
    output_md += f"- **Samples Scanned**: {result.get('scanned_count', 0)}\n"
    output_md += f"- **Uncertain Samples Found**: {result.get('uncertain_samples_found', 0)}\n\n"
    
    first_image_path = None
    first_text = ""
    
    if result.get('uncertain_samples_found', 0) > 0:
        samples = result.get('samples_to_review', [])
        output_md += "#### ⚠️ Top Uncertain Samples Requiring Human Review:\n"
        
        # 获取 Top 1 的图片路径和文本，以便在界面展示
        if len(samples) > 0:
            top_sample = samples[0]
            # 这里的 image 路径可能是相对路径 "data/xxx.jpg"，需要转换为绝对路径
            # 假设 config.SAMPLES_DIR 指向 "core/data/samples"
            # 而 dataset 中的路径是 "data/S04..."，我们需要去除前面的 "data/"
            img_rel_path = top_sample.get('image', '').replace('data/', '').replace('data\\', '')
            # 尝试在 core/data/samples/data 中找
            potential_path = os.path.join(SAMPLES_DIR, "data", img_rel_path)
            
            if os.path.exists(potential_path):
                first_image_path = potential_path
            else:
                 # 备选：直接拼接
                 first_image_path = os.path.join(SAMPLES_DIR, img_rel_path)
            
            first_text = top_sample.get('text', '')

        for i, sample in enumerate(samples[:5]): # 最多显示前5个
            output_md += f"**{i+1}. ID: {sample.get('id', 'unknown')}**\n"
            output_md += f"- Text: '{sample.get('text', '')}'\n"
            # output_md += f"- Image: {sample.get('image', '')}\n" # 图片路径太长，不显示在md里
            pred = sample.get('prediction', 'unknown')
            conf = sample.get('confidence', 0.0)
            ent = sample.get('entropy', 0.0)
            output_md += f"- Prediction: {pred} (Conf: {conf:.2f}, Entropy: {ent:.4f})\n\n"
            
    return output_md, first_image_path, first_text

# 构建 Gradio 界面
with gr.Blocks(title="Multimodal Conflict Inspector (MCP Client)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ Multimodal Conflict Inspector (MCP Client)")
    gr.Markdown("Based on BERT + CLIP + Cross-Attention | Powered by **MCP Protocol**")
    
    with gr.Tabs():
        # Tab 1: 单样本检测
        with gr.TabItem("🔍 Single Sample Inspection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="filepath", label="Upload Image")
                    input_text = gr.Textbox(lines=3, label="Text Description", placeholder="e.g., A happy family having dinner.")
                    inspect_btn = gr.Button("🚀 Run Inspection", variant="primary")
                
                with gr.Column(scale=1):
                    result_output = gr.Markdown(label="Prediction Result")
                    explanation_output = gr.Markdown(label="Attention Analysis (XAI)")
                    advisor_output = gr.Markdown(label="AI Advice")
                    
            with gr.Accordion("Raw MCP JSON Response", open=False):
                json_output = gr.Code(language="json", label="Protocol Data")
            
            inspect_btn.click(
                process_single_sample,
                inputs=[input_text, input_image],
                outputs=[result_output, explanation_output, advisor_output, json_output]
            )
            
        # Tab 2: 主动学习批量扫描
        with gr.TabItem("🔄 Active Learning Batch Scan"):
            gr.Markdown("### 🧠 Uncertainty Sampling Module")
            gr.Markdown("Automatically scan the dataset and identify high-entropy samples for human review.")
            
            with gr.Row():
                batch_slider = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Batch Size (Samples to Scan)")
                scan_btn = gr.Button("Start Batch Scan", variant="primary")
            
            # 新增：直接展示 Top 1 样本的详情，方便演示
            with gr.Row(visible=True):
                 with gr.Column(scale=1):
                     top_sample_image = gr.Image(label="Top 1 Uncertain Sample (Image)", interactive=False)
                 with gr.Column(scale=1):
                     top_sample_text = gr.Textbox(label="Top 1 Uncertain Sample (Text)", interactive=False, lines=4)
            
            scan_output = gr.Markdown(label="Scan Results List")
            
            scan_btn.click(
                run_active_learning,
                inputs=[batch_slider],
                outputs=[scan_output, top_sample_image, top_sample_text]
            )


    gr.Markdown("---")
    gr.Markdown("© 2026 YA Project | Multimodal Conflict Inspector Agent")

if __name__ == "__main__":
    # 允许 Gradio 自动选择可用端口 (server_port=None)
    demo.launch(share=False, server_name="127.0.0.1")