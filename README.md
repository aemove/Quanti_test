# Quanti_test: CNN Quantization & Training Experiments

这是一个基于 PyTorch 的模型训练与量化实验项目，使用 CIFAR-10 数据集。

目前正在实现VGG16的QAT与PTQ。

## 📂 项目结构

```
Quanti_test/
├── VGG16.py                    # 标准 VGG16 实现 
├── template.py                 # 模板脚本
├── VGG16_quanti/               # 主要实验目录
│   ├── VGG16_origin.py         # 原始 VGG16 训练脚本 (支持自动归档、Config配置)
│   ├── ptq_VGG16.py            # 训练后量化 (PTQ) 实验脚本
│   ├── ptq_VGG16_2.py          # 训练后量化 (PTQ) 实验脚本 2
│   ├── qat_VGG16.py            # 量化感知训练 (QAT) 实验脚本
│   └── record/                 # 实验结果记录 (自动生成)
│       └── result_YYYY-MM-DD_HH-MM/
│           ├── config.txt      # 实验参数记录
│           ├── best_model.ckpt # 最佳模型权重(已在 .gitignore 中忽略)
│           ├── acc_curve.png   # 准确率曲线
│           └── loss_curve.png  # 损失函数曲线
├── torchao_test/               # torchao 库测试目录
│   └── test.py
├── dataset/                    # 数据集目录 (已在 .gitignore 中忽略)
└── .gitignore                  # Git 忽略配置
```

## 🚀 功能特性

*   **VGG16 实现**：包含适配 CIFAR-10 (32x32) 的定制化 VGG16 模型。
*   **自动归档系统**：每次运行 `VGG16_quanti.py` 会自动在 `record/` 目录下生成带时间戳的文件夹，保存所有实验数据，方便回溯。
*   **配置管理**：通过 `Config` 类集中管理超参数（Learning Rate, Batch Size, Epochs 等）。
*   **可视化**：自动绘制并保存 Loss 和 Accuracy 训练曲线。
*   **Early Stopping**：支持早停机制，防止过拟合。

## 🏃‍♂️ 如何运行

1.  **准备数据集**：
    项目默认会在 `../dataset` 或 `./data` 目录下查找 CIFAR-10 数据集。如果不存在，脚本通常会自动下载。

2.  **运行训练脚本**：
    ```bash
    python VGG16_quanti/VGG16_quanti.py
    ```

3.  **查看结果**：
    运行结束后，前往 `VGG16_quanti/record/` 目录查看生成的最新实验文件夹。

## 📝 实验记录

所有的实验配置和结果都会自动保存在 `record` 文件夹中。例如：

*   `config.txt`: 记录了当次运行的所有参数（如 lr=0.0002, batch_size=128）。
*   `best_model.ckpt`: 验证集准确率最高的模型权重。

## ⚠️ 注意事项

*   **数据集与模型文件**：`dataset/` 文件夹和 `*.ckpt` 文件已被 `.gitignore` 忽略，不会上传到 GitHub。
