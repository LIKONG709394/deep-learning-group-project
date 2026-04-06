# INT4097 HW2 - Chinese Handwritten Character Recognition

> **Course**: INT4097 | **Student**: LI KONG (11603261) | **Seed**: 3261

---

## Overview

使用 CNN 对 Chinese MNIST 数据集（15 类汉字数字：零~九、十、百、千、万、亿）进行分类，要求验证准确率 ≥ 90%。

---

## A. Data Pipeline

### 数据来源

| 文件 | 内容 |
|------|------|
| `data.zip` | 15,000 张 64×64 灰度手写汉字图片 |
| `chinese_mnist.csv` | `suite_id`, `sample_id`, `code`, `value`, `character` 映射表 |

### 处理流程

```
data.zip → zipfile.extractall() → data_extracted/
                                        ↓
chinese_mnist.csv → pd.read_csv() → 构建 file_index 字典
                                        ↓
                              合并 image_path + code2label + label2char
                                        ↓
                        train_test_split(test_size=0.2, stratify)
                              ↓                    ↓
                        train_df (12000)      val_df (3000)
```

### 关键代码模式

- **文件名构建**: `input_{suite_id}_{sample_id}_{code}.jpg`
- **快速查找**: 先用 `os.walk` 建立 `file_index = {filename: fullpath}` 字典，再通过字典查找匹配 CSV 行
- **标签映射**: `code → label (int)` via `code2label`，`label → 汉字` via `label2char`

### Custom Dataset

```python
class ChineseMNISTDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('L')
        return self.transform(img), int(row['label'])
```

### DataLoader 参数

| 参数 | 值 |
|------|-----|
| `IMG_SIZE` | 64 |
| `BATCH_SIZE` | 128 |
| `shuffle` | True (train) / False (val) |
| Transform | Resize → ToTensor → Normalize([0.5], [0.5]) |

---

## B. CNN Architecture (ChineseCNN)

```
Input: [batch, 1, 64, 64]
   ↓
Conv2d(1→32, k=3, pad=1) → ReLU → MaxPool2d(2)    → [batch, 32, 32, 32]
   ↓
Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2)   → [batch, 64, 16, 16]
   ↓
Conv2d(64→128, k=3, pad=1) → ReLU → MaxPool2d(2)  → [batch, 128, 8, 8]
   ↓
Flatten                                              → [batch, 8192]
   ↓
Dropout(p) → Linear(8192→256) → ReLU → Dropout(p)  → [batch, 256]
   ↓
Linear(256→15)                                       → [batch, 15]
```

**关键设计**:
- 3 层 Conv2d + MaxPool2d 逐步提取特征
- Dropout 放在全连接层（flatten 后和 fc1 后各一次）
- `forward` 方法每层附带 shape trace 注释（Checkpoint 2 要求）

---

## C. Training & Optimization

### 主训练配置

| 参数 | 值 |
|------|-----|
| Optimizer | Adam, lr=5e-4 |
| Scheduler | StepLR(step_size=3, gamma=0.6) |
| Loss | CrossEntropyLoss |
| Epochs | 10 |
| Dropout | 0.3 |
| Best Val Acc | ~93% |

### 训练循环核心

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

- 使用 `copy.deepcopy(model.state_dict())` 保存最佳模型权重
- 训练结束后 `load_state_dict` 载回最佳权重

---

## D. Hyperparameter Comparison

| Config | LR | Dropout | 行为 | Best Val Acc |
|--------|------|---------|------|:----:|
| A (underfit) | 1e-5 | 0.7 | 学习率太小 + dropout 太高，几乎不学习，~6-7% | ~7% |
| B (diverge) | 0.01 | 0.0 | 学习率太大，梯度爆炸/发散，~6.7% | ~7% |
| **C (balanced)** | **5e-4** | **0.3** | **稳定收敛** | **~90%** |

### Failure Path 分析

- **Config A**: 欠拟合。LR=1e-5 太小 + Dropout=0.7 太高 → 模型无法学到任何东西，准确率停留在随机猜测水平
- **Config B**: 发散。LR=0.01 太大 → 梯度过大，优化器找不到有用方向，训练完全失败
- **修复**: 使用 Config C (lr=5e-4, dropout=0.3) 作为平衡配置

---

## E. Evaluation

### 混淆矩阵

- 使用 `sklearn.metrics.confusion_matrix` + `ConfusionMatrixDisplay`
- 标签格式: `index(汉字)`，如 `0(零)`, `1(一)` 等

### Classification Report

- `sklearn.metrics.classification_report` 输出每类 precision / recall / F1
- 整体 accuracy ~93%

### 最易混淆字符对

通过将混淆矩阵对角线置零后取 argmax 找出 Top-5 混淆对：
- 典型混淆: 千↔十（笔画相似）

---

## F. Anti-Plagiarism Checkpoints

| Checkpoint | 要求 | 实现 |
|:----:|------|------|
| 1 | 学生 ID 后四位作种子 | `STUDENT_ID_LAST_4 = 3261`，设置 `torch.manual_seed` + `np.random.seed` + `random.seed` + `cuda.manual_seed_all` |
| 2 | forward 方法 shape trace | 每层注释 `# -> [batch, C, H, W]` |
| 3 | 失败路径分析 | Config A (underfit) + Config B (diverge) 的分析与修复说明 |
| 4 | train/eval/no_grad 解释 | ~150 字英文段落 |

---

## G. Key Concepts

### `model.train()` vs `model.eval()`

| 模式 | Dropout | BatchNorm | 用途 |
|------|---------|-----------|------|
| `train()` | 随机丢弃 | 用 batch 统计量 | 训练时 |
| `eval()` | 全部保留 | 用 running 统计量 | 推理/验证时 |

### `torch.no_grad()`

- 关闭梯度追踪，节省显存，加速推理
- 与 `eval()` **独立** —— `eval()` 控制 Dropout/BN 行为，`no_grad()` 控制梯度计算
- 推理时两者都要用

---

## H. Project Structure

```
HW2_Chinese_Handwriting.ipynb
├── Cell 0      : 标题 + 学生信息
├── Cell 1      : Imports
├── Cell 3      : 中文字体设置 (Colab/本地)
├── Cell 5      : Checkpoint 1 - 种子设置
├── Cell 7-8    : 解压 data.zip + 检查
├── Cell 10-11  : CSV 读取 + 主查找表
├── Cell 13     : Train/Val split (80/20)
├── Cell 15-16  : Dataset + DataLoader
├── Cell 18     : 2×5 样本可视化
├── Cell 20     : CNN 定义 (Checkpoint 2)
├── Cell 22     : train_one_epoch / evaluate
├── Cell 24     : 主训练 (10 epochs)
├── Cell 26     : 主训练曲线
├── Cell 28-29  : 超参数比较 (3 configs × 5 epochs)
├── Cell 31     : 比较曲线 (2×2)
├── Cell 33     : 混淆矩阵
├── Cell 35     : Top-5 混淆对
├── Cell 37     : Classification Report
├── Cell 39     : 预测示例可视化
├── Cell 40     : Checkpoint 3 - 失败路径分析
├── Cell 41     : Checkpoint 4 - train/eval/no_grad
└── Cell 42     : Final Summary
```

---

## I. Scoring Checklist

| 类别 | 分值 | 状态 |
|------|:----:|:----:|
| 数据管道: Dataset/DataLoader + 10 个标注样本可视化 | 15 | ✅ |
| 模型设计: CNN + Dropout + Shape Trace | 20 | ✅ |
| 训练执行: 手动循环 + LR Scheduler + ≥90% acc | 20 | ✅ |
| 超参数比较: 3 组配置 + 结果记录 + 最佳配置 | 20 | ✅ |
| 检查点: ID 种子 + 失败实验 + train/eval 说明 | 15 | ✅ |
| 可视化: 损失曲线 + 混淆矩阵 + 混淆字符对 | 10 | ✅ |
| **Total** | **100** | **✅** |
