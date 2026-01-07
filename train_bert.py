import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from data_loader import load_dataset
import pandas as pd

import pandas as pd

def load_dataset(path, n_samples):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.head(n_samples)

    # 统一转成字符串 + 去空格 + 大写
    df["is_fraud_clean"] = (
        df["is_fraud"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 映射规则（非常宽容）
    label_map = {
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0,
        "YES": 1,
        "NO": 0,
        "是": 1,
        "否": 0,
        "诈骗": 1,
        "正常": 0,
        "NAN": None
    }

    df["label"] = df["is_fraud_clean"].map(label_map)

    # 丢弃无法识别的样本（这是科研里允许且常见的）
    before = len(df)
    df = df.dropna(subset=["label"])
    after = len(df)

    print(f"标签清洗：丢弃 {before - after} 条异常样本")

    texts = df["specific_dialogue_content"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels



class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 加载数据
texts, labels = load_dataset("训练集结果.csv", 1000)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
dataset = FraudDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=2
)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练 2 epoch 就够
model.train()
for epoch in range(2):
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        outputs.loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} finished")

model.save_pretrained("./bert_fraud")
tokenizer.save_pretrained("./bert_fraud")
