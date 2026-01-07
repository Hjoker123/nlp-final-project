# dydec.py
from dashscope import Generation
from keyword_analysis import extract_risk_keywords

def _build_instruction(keywords: str, round_id: int, prev_failed_reason: str = "") -> str:
    """
    生成动态改写指令（模拟 Red LLM）
    round_id: 第几轮
    prev_failed_reason: 上一轮失败原因（可为空）
    """
    base = (
        f"【第{round_id}轮】请针对风险点进行更隐蔽的改写。\n"
        f"风险关键词：{keywords}\n"
        "要求：\n"
        "1) 保持整体语义与对话意图一致；\n"
        "2) 弱化紧急/威胁/诱导语气，避免明显诈骗模式；\n"
        "3) 将命令式表达改为解释/建议式；\n"
        "4) 对话结构保持 left/right 格式，不要丢失关键信息。\n"
    )
    if prev_failed_reason:
        base += f"\n上一轮未成功原因提示：{prev_failed_reason}\n请据此进一步调整。"
    return base

def _attack_rewrite(dialogue_text: str, instruction: str) -> str:
    """
    执行改写（Attacking LLM）
    """
    prompt = f"""
你是对话改写助手。请严格按照“改写要求”改写“原始对话”，输出改写后的对话文本。

改写要求：
{instruction}

原始对话：
{dialogue_text}

改写后的对话：
"""
    resp = Generation.call(
        model="qwen-max",
        prompt=prompt,
        temperature=0.7,
        top_p=0.8
    )
    return resp.output.text

def dydec_attack_trace(
    dialogue_text: str,
    target_clf,
    max_iter: int = 5,
    threshold: float = 0.5,
    target_label: int | None = None,
) -> dict:
    """
    返回一个 trace 字典，包含每轮的 instruction / rewritten_text / pred_prob / pred_label
    target_clf: 你的 BertFraudClassifier（必须提供 predict_proba）
    target_label: 原始真实标签（如果提供可用于判定攻击成功）
    """
    keywords = extract_risk_keywords(dialogue_text)

    trace = {
        "keywords": keywords,
        "rounds": [],  # 每轮记录
        "final_text": None,
        "final_pred_prob": None,
        "final_pred_label": None,
        "iters_used": 0,
        "attack_success": None,
    }

    prev_failed_reason = ""

    current_text = dialogue_text
    for r in range(1, max_iter + 1):
        instruction = _build_instruction(keywords, r, prev_failed_reason)
        rewritten = _attack_rewrite(current_text, instruction)

        prob = float(target_clf.predict_proba([rewritten])[0])
        pred = int(prob > threshold)

        trace["rounds"].append({
            "round_id": r,
            "instruction": instruction,
            "rewritten_text": rewritten,
            "pred_prob": prob,
            "pred_label": pred
        })

        trace["iters_used"] = r
        trace["final_text"] = rewritten
        trace["final_pred_prob"] = prob
        trace["final_pred_label"] = pred

        # 判定是否“攻击成功”
        # 如果给了 target_label，则攻击成功 = pred != target_label
        # 如果没给 target_label，则默认以 “翻转预测” 作为成功（需要先算原始预测）
        if target_label is not None:
            if pred != int(target_label):
                trace["attack_success"] = True
                return trace
        else:
            # 无真实标签时：以翻转初始预测为成功
            if r == 1:
                # 第一次时计算原始预测（只算一次）
                orig_prob = float(target_clf.predict_proba([dialogue_text])[0])
                orig_pred = int(orig_prob > threshold)
                trace["orig_pred_prob"] = orig_prob
                trace["orig_pred_label"] = orig_pred
            if pred != trace["orig_pred_label"]:
                trace["attack_success"] = True
                return trace

        # 没成功：给下一轮一个“失败提示”，让指令更针对
        prev_failed_reason = (
            f"当前预测仍为 {pred}（prob={prob:.3f}），需要进一步降低模型对诈骗特征的敏感度。"
        )
        current_text = rewritten  # 下一轮基于上一轮继续改写

    trace["attack_success"] = False
    return trace
