# 🗑️TrashNet 垃圾分类 —— 基于 Swin Transformer 的图像分类

![Model](https://img.shields.io/badge/Model-Swin_Transformer-pink)
![Dataset](https://img.shields.io/badge/Dataset-TrashNet-orange)


本仓库提供了基于 [Swin Transformer](https://github.com/microsoft/Swin-Transformer) 的**垃圾图像分类**模型的完整训练与推理代码。
模型在 [TrashNet](https://github.com/garythung/trashnet) 数据集上进行微调，可识别六类常见垃圾——
**纸板、玻璃、金属、纸张、塑料和其他垃圾**，支持整批验证集评估与单张图片 Top-3 推理。

模型采用迁移学习策略，冻结主干网络前层权重，仅微调后三个 Block 及分类头，使用 NVIDIA RTX 4060GPU 训练 **20 个 epoch**，输入分辨率为 224×224，
在验证集上最终达到最佳准确率 **96.81%**。
本项目作为完整流程示例，涵盖数据加载、模型微调、训练监控与多场景推理的全过程。

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
├── weights/               # 模型权重（体积过大，已加入 .gitignore，不纳入版本控制）
├── tensorboard_logs/      # TensorBoard 日志（已加入 .gitignore，不纳入版本控制）
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


| 项目     | 配置                                                     |
| -------- | -------------------------------------------------------- |
| 基础模型 | Swin Transformer (swin_t)，ImageNet1K 预训练             |
| 微调策略 | 冻结前层，仅解冻 features[5~7]、norm 层及分类头          |
| 分类数   | 6（cardboard / glass / metal / paper / plastic / trash） |
| 输入尺寸 | 224 × 224                                               |
| 优化器   | AdamW，lr = 1e-4                                         |
| 损失函数 | CrossEntropyLoss                                         |
| 数据增强 | RandomHorizontalFlip、RandomRotation(15°)               |

---

## 训练结果


| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
| ----- | ---------- | -------- | --------- | ------- |
| 5     | 0.056      | 0.173    | 98.2%     | 95.2%   |
| 10    | 0.040      | 0.136    | 98.8%     | 96.8%   |
| 15    | 0.028      | 0.249    | 99.3%     | 94.0%   |
| 20    | 0.009      | 0.165    | 99.7%     | 95.6%   |

**最佳验证准确率：96.81%**
