# ablation_synonym_llm.py
import re
from dashscope import Generation

def synonym_replace_llm(text: str, max_replacements: int = 3) -> str:
    """
    用 Qwen 做“词级同义替换”：
    - 只允许替换少量关键词/短语
    - 不允许改句式、不允许增删句子
    - 尽量保持原文结构（尤其 left/right 格式）
    """
    prompt = f"""
你是文本编辑器。请对下面对话做“同义词替换”消融改写，严格遵守规则：
1) 只允许做【同义词/近义短语替换】，不要改写整句，不要改语序，不要增删句子。
2) 保持 left:/right: 行结构不变，不要合并行。
3) 最多替换 {max_replacements} 处（少量即可）。
4) 优先替换带有强诱导/强风险的词，如：立即/点击/链接/验证/异常/冻结/收益/投资/客服 等。
5) 输出只包含改写后的对话文本，不要解释。

原始对话：
{text}

改写后的对话：
"""
    resp = Generation.call(
        model="qwen-plus",   # 消融实验用 plus 就行，省额度
        prompt=prompt,
        temperature=0.3,     # 低一点，减少“乱改句子”
        top_p=0.8
    )
    out = resp.output.text.strip()

    # 保险：如果模型乱输出带“解释/规则”，简单截取最后一段
    # （一般不会触发，但加上更稳）
    return out
