# data_loader.py
import pandas as pd

def load_dataset(path, n_samples=200, seed=42):
    df = pd.read_csv(path, encoding="utf-8-sig")

    # 1) 清洗列名：去空格、去不可见字符
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # 2) 自动识别文本列
    text_candidates = ["specific_dialogue_content", "dialogue", "text", "content"]
    text_col = None
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"找不到文本列，当前列名为：{list(df.columns)}")

    # 3) 自动识别标签列
    label_candidates = ["is_fraud", "label", "y", "target"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"找不到标签列，当前列名为：{list(df.columns)}")

    # 4) 取样（建议随机取，避免前200条全同一类）
    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    # 5) 清洗标签值并映射
    s = df[label_col].astype(str).str.strip().str.upper()

    label_map = {
        "TRUE": 1, "FALSE": 0,
        "1": 1, "0": 0,
        "YES": 1, "NO": 0,
        "T": 1, "F": 0,
        "是": 1, "否": 0,
        "诈骗": 1, "正常": 0,
        # ✅ 关键：把 NAN 当缺失
        "NAN": None, "NONE": None, "": None
    }

    df["label"] = s.map(label_map)

    # ✅ 关键：丢弃 label 为空的行（NaN 标签）
    before = len(df)
    df = df.dropna(subset=["label"])
    after = len(df)
    print(f"标签清洗：丢弃 {before - after} 条缺失/异常标签样本")

    texts = df[text_col].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels
