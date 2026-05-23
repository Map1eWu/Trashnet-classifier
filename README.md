# TrashNet 垃圾分类 —— 基于 Swin Transformer 的图像分类

基于 Swin Transformer 对 [TrashNet](https://github.com/garythung/trashnet) 数据集进行微调，实现六类垃圾的自动分类，最佳验证准确率达 **96.81%**。

---

## 项目结构

```
Trashnet-classifier/
├── initialize.py          # 数据集加载与 DataLoader 构建
├── train_trashnet.py      # 训练脚本（含日志、TensorBoard、断点续训）
├── val_trashnet.py        # 验证 & 单图推理脚本
├── log/
│   ├── train.log          # 训练过程文本日志
│   └── metrics.csv        # 逐步指标记录
├── test_img/              # 示例测试图片
├── weights/               # 模型权重（不纳入版本控制）
└── .gitignore
```

---

## 环境依赖

```bash
pip install torch torchvision tqdm tensorboard pillow
```

---

## 数据集准备

从 [TrashNet](https://github.com/garythung/trashnet) 下载数据集，按以下结构划分：

```
dataset-split/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── val/
    └── ...（同上）
```

---

## 训练

```bash
python train_trashnet.py \
    --dataset_train_path /path/to/dataset-split/train \
    --dataset_val_path   /path/to/dataset-split/val \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --device cuda \
    --output_path ./run/
```

**断点续训：**

```bash
python train_trashnet.py \
    --resume_weights ./run/weights/latest_model.pth \
    ...
```

---

## 验证 & 推理

**整个验证集评估：**

```bash
python val_trashnet.py \
    --dataset_val_path /path/to/dataset-split/val \
    --weights_path ./run/weights/best_model.pth \
    --device cuda
```

**单张图片推理（输出 Top-3 预测）：**

```bash
python val_trashnet.py \
    --img_path ./test_img/test_img.jpg \
    --weights_path ./run/weights/best_model.pth \
    --device cuda
```

---

## 模型设计

| 项目 | 配置 |
|------|------|
| 基础模型 | Swin Transformer (swin_t)，ImageNet1K 预训练 |
| 微调策略 | 冻结前层，仅解冻 features[5~7]、norm 层及分类头 |
| 分类数 | 6（cardboard / glass / metal / paper / plastic / trash） |
| 输入尺寸 | 224 × 224 |
| 优化器 | AdamW，lr = 1e-4 |
| 损失函数 | CrossEntropyLoss |
| 数据增强 | RandomHorizontalFlip、RandomRotation(15°) |

---

## 训练结果

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 15    | 0.028     | 0.249    | 99.3%     | 94.0%   |
| 16    | 0.057     | 0.210    | 98.1%     | 96.0%   |
| 19    | 0.019     | 0.177    | 99.4%     | 96.4%   |
| 20    | 0.009     | 0.165    | 99.7%     | 95.6%   |

**最佳验证准确率：96.81%**（best_model.pth）

---

## 查看 TensorBoard

```bash
tensorboard --logdir ./run/tensorboard_logs/
```
