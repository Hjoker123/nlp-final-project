import os
import pandas as pd
import numpy as np
from collections import Counter

from data_loader import load_dataset
from bert_classifier import BertFraudClassifier
from ablation_synonym_llm import synonym_replace_llm

RAW_PATH = "测试集结果.csv"
N_SAMPLES = 200

OUT_CSV = "ablation_synonym_llm_200.csv"
THRESH = 0.5

def load_or_init(texts, labels):
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
        if len(df) == len(texts):
            return df
        # 长度不一致说明你换了数据或采样变了，重置
        os.remove(OUT_CSV)

    df = pd.DataFrame({
        "idx": list(range(len(texts))),
        "original_text": texts,
        "label": labels,
        "adv_text": ["" for _ in range(len(texts))],
        "pred_prob": [None for _ in range(len(texts))],
        "pred_label": [None for _ in range(len(texts))],
    })
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return df

def main():
    texts, labels = load_dataset(RAW_PATH, N_SAMPLES)
    labels = np.array(labels)

    clf = BertFraudClassifier("./bert_fraud")
    df = load_or_init(texts, labels)

    pending = df[df["adv_text"].isna() | (df["adv_text"].astype(str).str.strip() == "")]
    print(f"LLM 同义替换断点续跑：已完成 {len(df)-len(pending)}/{len(df)}")

    # ===== 逐条生成（断点续跑）=====
    for _, row in pending.iterrows():
        idx = int(row["idx"])
        original = row["original_text"]
        print(f"同义替换（LLM）处理中 {idx+1}/{len(df)} (idx={idx})")

        try:
            adv = synonym_replace_llm(original, max_replacements=3)
        except Exception as e:
            print(f"⚠️ idx={idx} 生成失败：{e}，跳过（下次继续）")
            continue

        df.loc[df["idx"] == idx, "adv_text"] = adv
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ===== 统一评估并写回预测结果 =====
    ok = ~(df["adv_text"].isna() | (df["adv_text"].astype(str).str.strip() == ""))
    eval_df = df[ok].copy()
    adv_texts = eval_df["adv_text"].astype(str).tolist()

    probs = clf.predict_proba(adv_texts)
    preds = (probs > THRESH).astype(int)

    df.loc[eval_df.index, "pred_prob"] = probs
    df.loc[eval_df.index, "pred_label"] = preds
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    y_true = eval_df["label"].astype(int).to_numpy()
    acc = (preds == y_true).mean()

    print("【实验4.4】LLM 同义替换准确率:", float(acc))
    print("y_true 分布:", Counter(y_true.tolist()))
    print("y_pred 分布:", Counter(preds.tolist()))
    print(f"已保存到: {OUT_CSV}")

if __name__ == "__main__":
    main()
