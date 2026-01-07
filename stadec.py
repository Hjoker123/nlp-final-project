# stadec.py
from dashscope import Generation

def stadec_rewrite(dialogue_text: str) -> str:
    """
    StaDec：使用固定提示进行对抗改写
    不分析原因，直接尝试欺骗分类器
    """
    prompt = f"""
请改写以下对话文本，保持原有语义不变，
但尽量弱化其中可能被识别为诈骗或诱导的表达方式，
使其更接近正常客服或日常沟通。

原始对话：
{dialogue_text}

改写后的对话：
"""

    response = Generation.call(
        model="qwen-plus",
        prompt=prompt,
        temperature=0.7,
        top_p=0.8
    )

    return response.output.text
