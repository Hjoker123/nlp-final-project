# bert_classifier.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BertFraudClassifier:
    def __init__(self, model_path: str = "./bert_fraud"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # 如果你有 GPU，这行会更快；没有也没关系
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict_proba(self, texts, max_length: int = 256):
        """
        返回每条文本属于 fraud(=1) 的概率，shape: [N]
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # fraud 概率
        return probs.detach().cpu().numpy()

    # ✅ 给你兼容你原来写法：clf.predict(texts)
    def predict(self, texts, threshold: float = 0.5):
        probs = self.predict_proba(texts)
        return (probs > threshold).astype(int)
