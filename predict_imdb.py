import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 1. 重建和训练时完全一致的模型结构 ----------
VOCAB_SIZE = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased").vocab_size
EMBED_DIM = 64

class SimpleTextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(VOCAB_SIZE, EMBED_DIM, mode="mean")
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        return self.fc(emb)

# ---------- 2. 加载最佳模型权重 ----------
model = SimpleTextClassifier().to(device)
model.load_state_dict(torch.load('best_imdb.pt', map_location=device))
model.eval()

# ---------- 3. 分词器 ----------
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# ---------- 4. 推理函数 ----------
def predict(text):
    text = text.strip()
    if len(text.split()) < 5:
        return "输入太短（少于 5 个词），模型只对完整影评可靠", 0.0
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    label = "正面 (Positive)" if pred == 1 else "负面 (Negative)"
    return label, confidence

# ---------- 5. 交互式测试 ----------
if __name__ == "__main__":
    print("IMDB 情感分类推理脚本（输入 q 退出）")
    print("-" * 50)
    while True:
        text = input("\n输入影评: ").strip()
        if text.lower() == 'q':
            break
        if not text:
            continue
        label, conf = predict(text)
        print(f"预测: {label} | 置信度: {conf:.4f}")
