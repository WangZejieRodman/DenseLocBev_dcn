# DenseLocBEV: 基于 Dense BEV 与可变形卷积的巷道场景识别

本项目实现了一个针对地下矿井巷道环境的 3D LiDAR 点云场景识别（Place Recognition）系统。

**核心理念**：本项目摒弃了传统的稀疏体素化（Sparse Voxelization）或直接点云处理方法，而是采用了 **Dense BEV (Bird's Eye View)** 编码逻辑。通过将 3D 点云在 Z 轴（高度）上的信息进行量化并“压扁”，将其转化为特征通道（Channels），从而将 3D 点云转化为一张“多通道的 2D 图像”。这使得我们可以利用成熟且高效的 2D CNN（如 ResNet）配合可变形卷积（DCN）来处理巷道中的几何特征。

## ✨ 主要特性

* **Dense BEV 编码**: 将点云空间 `(X, Y, Z)` 映射为张量 `(C, H, W)`。其中  为水平面的网格， 为高度方向的切片层数（Z-axis slicing）。
* **Next-Gen Backbone**:
* **ECA (Efficient Channel Attention)**: 在浅层引入通道注意力机制，用于抑制矿井底板和顶板的噪声，增强侧壁特征。
* **DCNv2 (Deformable Convolution)**: 在深层网络中使用可变形卷积，自适应地提取弯曲巷道的几何特征。


* **NetVLAD 聚合**: 使用 NetVLAD 生成全局描述子，支持端到端训练。
* **鲁棒性评估**: 内置旋转不变性评估脚本，测试模型在 0° 到 180° 旋转下的检索性能。
* **损失函数**: 支持 Truncated SmoothAP Loss，优化检索排序。

## 🛠️ 环境依赖

* Python 3.8+
* PyTorch (建议 1.10+, 需支持 CUDA)
* Torchvision
* NumPy, Pandas, Scikit-learn
* OpenCV (可选，用于可视化)

虽然代码中包含 MinkowskiEngine 的引用（遗留代码），但在 DenseBEV 模式下，核心模型仅依赖标准的 PyTorch 操作。

```bash
pip install torch torchvision numpy pandas scikit-learn tqdm configparser

```

## 📂 数据准备

本项目使用 **Chilean Underground Mine Dataset**。请确保数据按以下结构存放，并根据实际路径修改配置文件。

### 1. 生成查询字典

运行以下脚本将原始点云数据划分为训练集和测试集索引（Pickle 文件）。

```bash
# 生成训练集的 Positives/Non-negatives 字典 (Session 100-159)
python datasets/chilean/generate_training_tuples_chilean.py

# 生成评估用的 Database/Query 字典 (Database: 160-189, Query: 195-209)
python datasets/chilean/generate_test_sets_chilean.py

```

*注意：请修改脚本中的 `BASE_PATH` 变量为您本地的数据集路径。*

## 🚀 训练模型

训练脚本会自动加载配置，训练模型，并在训练结束后进行旋转鲁棒性评估。

### 配置文件

核心配置位于 `config/config_chilean_bev.txt`。关键参数包括：

* `batch_size`: 建议尽可能大（如 128），受显存限制。
* `div_n`: BEV 网格划分，默认为 `256, 256, 32` (对应 X, Y, Z)。
* `loss`: 默认为 `TruncatedSmoothAP`。

### 启动训练

建议在 IDE (如 PyCharm) 中运行，或通过命令行：

```bash
export PYTHONPATH=.
python training/train_chilean_bev.py

```

训练过程中，日志将保存在 `training/trainer.log`，权重保存在 `weights/` 目录下。

## 📊 评估与测试

### 旋转鲁棒性评估

模型在训练结束后会自动运行评估。如果需要单独评估已保存的模型权重，请运行：

```bash
python eval/evaluate_chilean_rotation.py

```

**注意**：运行前请在脚本中修改 `args.weights` 指向您要测试的 `.pth` 模型文件路径。

该脚本会测试 Query 点云旋转 `[0, 5, 10, 15, 30, 45, 60, 90, 135, 180]` 度时的 Recall@1 和 Recall@1% 性能，并生成分析报告。

## 🏗️ 模型架构详解

### 1. 输入量化 (Quantization)

`datasets/quantization.py` 中的 `BEVQuantizer` 将点云转换为 Dense Tensor：

* **输入**:  点云 
* **处理**: 划分为  的网格，高度方向划分为 32 层。
* **输出**:  的张量。此时，**高度 Z 变成了特征通道 C**。

### 2. 骨干网络 (Backbone)

位于 `models/densebev.py`，基于 ResNet 结构改进：

* **Stem**: 7x7 卷积 + **DenseECALayer** (通道注意力)。
* **Stage 1-2**: 标准 ResBlock。
* **Stage 3-4**: 集成 **DCNv2Block** (可变形卷积)，适应巷道的不规则形状。
* **输出**:  的特征图。

### 3. 聚合与输出 (Aggregation)

* **Pooling**: `models/layers/pooling_wrapper_dense.py` 封装了 NetVLAD。
* **NetVLAD**: 将特征图展平并聚类，输出最终的全局描述子（如 256 维）。

## 📁 目录结构

```text
.
├── config/                 # 训练配置文件
├── datasets/               # 数据加载与预处理
│   ├── chilean/            # Chilean 数据集专用脚本
│   ├── quantization.py     # 核心：BEV 量化器
│   └── augmentation.py     # 数据增强
├── eval/                   # 评估脚本
│   ├── evaluate_chilean.py
│   └── evaluate_chilean_rotation.py
├── models/                 # 模型定义
│   ├── denseloc.py         # 模型主类
│   ├── densebev.py         # Backbone (ResNet+DCN+ECA)
│   ├── layers/             # 网络层 (NetVLAD, DCN, ECA等)
│   └── losses/             # 损失函数 (SmoothAP, Triplet等)
├── training/               # 训练逻辑
│   ├── trainer.py          # 训练循环
│   └── train_chilean_bev.py # 训练入口脚本
└── misc/                   # 工具函数

```

## 📝 引用与致谢

本项目基于 Warsaw University of Technology 的 MinkLoc 系列工作进行改进，针对矿井巷道环境进行了 Dense BEV 适配。

核心参考论文：

* *MinkLoc3D: Point Cloud Based Large-Scale Place Recognition*
* *Recall@k Surrogate Loss with Large Batches and Similarity Mixup*
