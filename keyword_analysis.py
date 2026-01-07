# keyword_analysis.py

def extract_risk_keywords(text: str) -> str:
    """
    从对话文本中提取常见诈骗风险关键词
    用于模拟 Reasoning LLM 的分析能力
    """
    risk_words = [
        "异常", "验证", "链接", "点击", "立即",
        "账户", "账号", "冻结", "风险",
        "投资", "收益", "回报", "高收益",
        "客服", "系统", "短信", "通知"
    ]

    found = []
    for w in risk_words:
        if w in text:
            found.append(w)

    if not found:
        return "无明显高风险关键词"
    return "、".join(found)
