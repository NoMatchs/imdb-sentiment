# IMDB Sentiment Classifier

基于 PyTorch EmbeddingBag 的 IMDB 影评情感分类。

## 文件
- `train_demo6.py`：训练脚本（IMDB 2000 条，早停，保存 best_imdb.pt）
- `predict_imdb.py`：推理脚本，输入完整影评输出正面/负面

## 运行
pip install torch transformers datasets
python train_demo6.py
python predict_imdb.py
