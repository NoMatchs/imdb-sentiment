import os
# 复用你之前解决 401 的经验：走镜像 + 禁用 xet
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# 1. 加载 IMDB（先只取一部分，避免跑太久）
raw = load_dataset("stanfordnlp/imdb")
# 为了快速验证，训练集取 2000 条，测试集取 500 条
train_data = raw["train"].shuffle(seed=42).select(range(2000))
test_data = raw["test"].shuffle(seed=42).select(range(500))

# 2. 用 DistilBERT 的分词器把文本转成 token id
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

train_data = train_data.map(tokenize_fn, batched=True)
test_data = test_data.map(tokenize_fn, batched=True)

# 只保留模型需要的列，并把 label 改名为 labels
train_data = train_data.remove_columns(["text"]).rename_column("label", "labels")
test_data = test_data.remove_columns(["text"]).rename_column("label", "labels")
train_data.set_format("torch")
test_data.set_format("torch")

# 3. DataLoader（用 DataCollatorWithPadding 做动态 padding）
collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collator)
val_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collator)

# 4. 一个简单的 EmbeddingBag 分类器（延续你之前 nn.Embedding 的思路）
# 这里直接用随机初始化的 Embedding，不加载预训练词向量，保持轻量
VOCAB_SIZE = tokenizer.vocab_size
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
        # EmbeddingBag 需要 flatten 的 token ids 和 offsets，
        # 但这里为了简化，直接对 embedding 结果做平均池化
        emb = self.embedding(input_ids)  # (batch, embed_dim)，mode="mean" 自动处理
        return self.fc(emb)

model = SimpleTextClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练 + 验证（沿用你之前的骨架）
max_epochs, patience = 10, 3
best_val_acc, no_improve = 0.0, 0

for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total

    if val_acc > best_val_acc:
        best_val_acc, no_improve = val_acc, 0
        torch.save(model.state_dict(), 'best_imdb.pt')
        print(f"epoch {epoch:2d} | val acc {val_acc:.4f} *best*")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"早停于 epoch {epoch}，最佳 val acc = {best_val_acc:.4f}")
            break
