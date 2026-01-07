# exp_1_original.py
from data_loader import load_dataset
from bert_classifier import BertFraudClassifier
import numpy as np

texts, labels = load_dataset("测试集结果.csv", 200)
clf = BertFraudClassifier("./bert_fraud")  # 确保加载的是你训练后的模型

preds = clf.predict(texts, threshold=0.5)
labels = np.array(labels)

acc = (preds == labels).mean()
print("【实验4.1】原始数据集准确率:", acc)

