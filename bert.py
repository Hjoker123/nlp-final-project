# data_loader.py
import pandas as pd

def load_dataset(path, n_samples=200):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.head(n_samples)
    df["label"] = df["is_fraud"].map({"TRUE": 1, "FALSE": 0})
    texts = df["specific_dialogue_content"].tolist()
    labels = df["label"].tolist()
    return texts, labels
# bert_classifier.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BertFraudClassifier:
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.model.eval()

    def predict(self, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1].numpy()  # fraud 概率
