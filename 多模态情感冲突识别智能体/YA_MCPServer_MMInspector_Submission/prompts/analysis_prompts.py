from . import YA_MCPServer_Prompt

@YA_MCPServer_Prompt(name="analyze_conflict", description="Multimodal conflict analysis prompt.")
def analyze_conflict(text: str, image_description: str) -> str:
    """
    提供多模态冲突分析的 LLM 指令模板。
    
    Args:
        text (str): 输入的文本内容。
        image_description (str): 图像内容的简要描述。
        
    Returns:
        str: 完整的 Promt 文本，用于指导 LLM 分析图文冲突类型。
    """
    prompt_template = f"""
    You are an expert in multimodal conflict detection. Please analyze the following text and image description:
    
    Text: "{text}"
    Image: "{image_description}"
    
    Task:
    1. Determine if there is a semantic conflict between the text and the image.
    2. Classify the conflict type (e.g., Sarcasm, Irony, Contradiction).
    3. Explain your reasoning based on emotional tone and factual consistency.
    
    Output Format:
    - Conflict Exists: Yes/No
    - Conflict Type: [Type]
    - Explanation: [Reasoning]
    """
    return prompt_template