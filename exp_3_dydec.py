import os
import json
import pandas as pd
import numpy as np
from collections import Counter

from data_loader import load_dataset
from bert_classifier import BertFraudClassifier
from dydec import dydec_attack_trace

RAW_PATH = "测试集结果.csv"
N_SAMPLES = 200

OUT_JSONL = "dydec_trace_200.jsonl"   # 每行一个样本的完整 trace
OUT_CSV = "dydec_summary_200.csv"     # 展示用汇总表（方便贴报告）

MAX_ITER = 5
THRESH = 0.5


def load_done_indices(jsonl_path):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done.add(int(obj["idx"]))
    return done


def append_jsonl(jsonl_path, obj):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    texts, labels = load_dataset(RAW_PATH, N_SAMPLES)
    labels = np.array(labels)

    clf = BertFraudClassifier("./bert_fraud")

    done = load_done_indices(OUT_JSONL)
    print(f"DyDec 断点续跑：已完成 {len(done)}/{len(texts)}")

    # 用于最后生成 summary
    summary_rows = []

    for idx, (t, y) in enumerate(zip(texts, labels)):
        if idx in done:
            continue

        print(f"DyDec 处理中 {idx+1}/{len(texts)} (idx={idx})")

        trace = dydec_attack_trace(
            dialogue_text=t,
            target_clf=clf,
            max_iter=MAX_ITER,
            threshold=THRESH,
            target_label=int(y)  # 用真实标签判断攻击成功
        )

        record = {
            "idx": idx,
            "label": int(y),
            "original_text": t,
            **trace
        }
        append_jsonl(OUT_JSONL, record)

    # ===== 生成汇总 CSV（每次跑完都重新读一遍 jsonl 汇总，最简单稳健）=====
    if not os.path.exists(OUT_JSONL):
        print("没有生成任何 trace，退出。")
        return

    # 读取所有结果
    all_records = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # 汇总字段（适合写报告表格）
    for obj in all_records:
        summary_rows.append({
            "idx": obj["idx"],
            "label": obj["label"],
            "iters_used": obj.get("iters_used", None),
            "attack_success": obj.get("attack_success", None),
            "final_pred_prob": obj.get("final_pred_prob", None),
            "final_pred_label": obj.get("final_pred_label", None),
            "keywords": obj.get("keywords", ""),
        })

    df_sum = pd.DataFrame(summary_rows).sort_values("idx")
    df_sum.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    avg_iters = df_sum["iters_used"].mean()
    print("DyDec 平均迭代次数:", float(avg_iters))

    # ===== 计算 DyDec 后的准确率（用 final_pred_label vs label）=====
    y_true = df_sum["label"].astype(int).to_numpy()
    y_pred = df_sum["final_pred_label"].astype(int).to_numpy()
    acc = (y_true == y_pred).mean()

    print("【实验4.3】DyDec 后准确率:", float(acc))
    print("y_true 分布:", Counter(y_true.tolist()))
    print("y_pred 分布:", Counter(y_pred.tolist()))
    print("攻击成功率(ASR):", float(df_sum["attack_success"].astype(int).mean()))
    print(f"已保存：\n- 详细 trace: {OUT_JSONL}\n- 汇总表: {OUT_CSV}")


if __name__ == "__main__":
    main()
