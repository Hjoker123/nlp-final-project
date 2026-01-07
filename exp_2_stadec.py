import os
import pandas as pd
import numpy as np
from collections import Counter

from data_loader import load_dataset
from bert_classifier import BertFraudClassifier
from stadec import stadec_rewrite


RAW_PATH = "测试集结果.csv"
N_SAMPLES = 200

OUT_CSV = "stadec_adv_200.csv"          # 保存改写结果
OUT_PROGRESS = "stadec_progress.txt"    # 保存进度（可选）


def load_or_init_table(texts, labels, out_csv):
    """
    如果已有输出CSV，就读入并用于断点续跑；
    否则初始化一个包含原文、标签、改写列的表。
    """
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv, encoding="utf-8-sig")
        # 基本校验：长度不一致就重新初始化（避免你换了数据还接着跑）
        if len(df) != len(texts):
            print("检测到输出文件与当前样本数不一致，将重新生成输出文件。")
            os.remove(out_csv)
        else:
            return df

    df = pd.DataFrame({
    "idx": list(range(len(texts))),
    "original_text": texts,
    "label": labels,              # 原始真实标签
    "adv_text": ["" for _ in range(len(texts))],
    "pred_prob": [None for _ in range(len(texts))],   # 新增
    "pred_label": [None for _ in range(len(texts))],  # 新增
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


def main():
    texts, labels = load_dataset(RAW_PATH, N_SAMPLES)
    clf = BertFraudClassifier("./bert_fraud")  # 确保加载训练后的模型

    df = load_or_init_table(texts, labels, OUT_CSV)

    # 找出未完成的样本（adv_text 为空或 NaN）
    pending_mask = df["adv_text"].isna() | (df["adv_text"].astype(str).str.strip() == "")
    pending_indices = df.loc[pending_mask, "idx"].tolist()

    done = len(df) - len(pending_indices)
    print(f"StaDec 断点续跑：已完成 {done}/{len(df)}，待处理 {len(pending_indices)} 条。")

    # 逐条改写，并实时落盘
    for k, idx in enumerate(pending_indices, start=1):
        original = df.loc[df["idx"] == idx, "original_text"].values[0]
        print(f"StaDec 处理中 idx={idx}（剩余 {len(pending_indices)-k+1} 条）")

        try:
            adv = stadec_rewrite(original)
        except Exception as e:
            print(f"⚠️ idx={idx} 调用失败：{e}，跳过（下次可继续跑）")
            continue

        # 写回表格并保存（每条保存一次，最稳）
        df.loc[df["idx"] == idx, "adv_text"] = adv
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

        # 可选：保存进度文件（便于你截图/记录）
        with open(OUT_PROGRESS, "w", encoding="utf-8") as f:
            f.write(f"done={idx}\n")

    # ===== 生成完后做评估 =====
    ok_mask = ~(df["adv_text"].isna() | (df["adv_text"].astype(str).str.strip() == ""))
    eval_df = df.loc[ok_mask].copy()

    if len(eval_df) == 0:
        print("没有任何样本成功生成，无法评估。")
        return

    adv_texts = eval_df["adv_text"].astype(str).tolist()
    y_true = eval_df["label"].astype(int).to_numpy()

    # 预测
    probs = clf.predict_proba(adv_texts)
    y_pred = (probs > 0.5).astype(int)

    # ===== 写回预测结果（关键新增部分）=====
    df.loc[eval_df.index, "pred_prob"] = probs
    df.loc[eval_df.index, "pred_label"] = y_pred

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ===== 打印评估指标 =====
    acc = (y_pred == y_true).mean()
    print("【实验4.2】StaDec 后准确率:", float(acc))



if __name__ == "__main__":
    main()
