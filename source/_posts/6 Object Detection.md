---
categories:
- 学习笔记
date: '2025-11-30 14:35:20'
title: 6 Object Detection
---
# Lecture 22: Object Detection (目标检测)

## 1. 任务定义与挑战

目标检测的任务不仅是判断图像中有什么（分类），还要确定它们在哪里（定位）。

### 1.1 任务定义
* **输入**：单张 RGB 图像。
* **输出**：一组检测到的物体 $\{ (c_i, b_i) \}$，其中：
    * $c_i$ 是类别标签（如“猫”、“车”）。
    * $b_i$ 是边界框 (Bounding Box)，通常表示为 $(x, y, w, h)$，即中心坐标及宽高。

### 1.2 核心挑战
* **输出数量不固定**：图像中可能包含 0 到任意多个物体，无法像分类任务那样输出固定维度的向量。
* **多任务学习**：需要同时进行分类（离散）和回归（连续）。
* **尺度变化**：物体在图像中的尺寸差异巨大。

### 1.3 评估指标
* **IoU (Intersection over Union)**：交并比，用于衡量预测框与真实框 (Ground Truth) 的重叠程度。

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

* **mAP (mean Average Precision)**：各类别的平均精度 (AP) 的均值，是衡量检测器性能的黄金标准。

---

## 2. 传统方法：滑动窗口 (Sliding Window)

最直观的方法是使用不同大小和长宽比的窗口在图像上滑动，将每个窗口内的图像块送入分类器。
* **缺点**：计算开销巨大。绝大多数窗口都是背景，且为了覆盖不同位置和尺度，窗口数量呈指数级增长，效率极低。

---

## 3. 两阶段检测器 (Two-Stage Detectors)

这类算法将检测过程分为两步：(1) 生成可能包含物体的候选区域 (Region Proposals)；(2) 对这些区域进行分类和精修。

### 3.1 R-CNN (Regions with CNN features)
深度学习目标检测的开山之作。
* **流程**：
    1.  使用 **Selective Search** 算法在图像上提取约 2000 个候选区域 (Proposals)。
    2.  将每个候选区域缩放 (Warp) 到固定大小，输入 CNN 提取特征。
    3.  使用 SVM 进行分类，使用线性回归器微调边界框。
* **缺点**：速度慢（每个框都要过一遍 CNN），训练繁琐（多阶段流水线）。

### 3.2 Fast R-CNN
针对 R-CNN 的速度瓶颈进行了改进。
* **核心创新**：
    1.  **特征共享**：整张图像只过一次 CNN，得到特征图 (Feature Map)。
    2.  **RoI Pooling**：在特征图上根据候选框的位置切取特征，并将不同尺寸的区域映射为固定尺寸的特征向量，从而可以接入全连接层。
    3.  **多任务损失**：分类和回归同时训练。
* **局限**：候选区域的生成仍然依赖 CPU 上的 Selective Search 算法，成为速度瓶颈。

### 3.3 Faster R-CNN
引入了 **RPN (Region Proposal Network)**，实现了真正的端到端训练。
* **RPN**：一个全卷积网络，用于在特征图上生成候选框。
    * **锚点 (Anchors)**：在特征图的每个位置预设 $k$ 个不同尺度和比例的参考框。RPN 预测这些锚点是前景还是背景，以及它们的坐标偏移。
* **流程**：图像 $\to$ CNN Backbone $\to$ RPN 生成 Proposals $\to$ RoI Pooling $\to$ 分类与回归。

---

## 4. 单阶段检测器 (Single-Stage Detectors)

单阶段检测器没有显式的候选区域生成步骤，直接在特征图上进行密集的分类和回归。

### 4.1 YOLO & SSD
* **特点**：速度极快，将检测问题转化为回归问题。直接在网格上预测边界框和类别概率。
* **早期问题**：精度通常低于两阶段检测器，主要受限于正负样本极端不平衡（背景框太多）。

### 4.2 RetinaNet 与 Focal Loss
RetinaNet 的提出缩小了单阶段与两阶段检测器的精度差距。
* **FPN (Feature Pyramid Network)**：构建多尺度特征金字塔，利用顶层的语义信息和底层的高分辨率信息，在不同层级检测不同大小的物体。
* **Focal Loss**：一种动态调整权重的损失函数，降低易分类样本（背景）的权重，使模型专注于难分类样本的训练。

$$
\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)
$$

---

## 5. 基于 Transformer 的检测器：DETR

DETR (DEtection TRansformer) 将目标检测重新建模为**集合预测 (Set Prediction)** 问题，摒弃了 Anchor 和 NMS 等手工设计的组件。

### 5.1 架构
1.  **CNN Backbone**：提取图像特征。
2.  **Transformer Encoder-Decoder**：利用 Self-Attention 处理全局上下文。
3.  **Object Queries**：Decoder 输入一组可学习的查询向量 (Queries)，每个 Query 负责预测一个物体（或“无物体”）。

### 5.2 二分图匹配 (Bipartite Matching)
训练时，使用匈牙利算法将 $N$ 个预测框与 $M$ 个真实框进行一对一的最优匹配（Ground Truth 不足的部分用“空物体”补齐），计算损失函数。这使得模型可以直接输出最终的检测集合，无需后处理。

---

## 6. 实例分割 (Instance Segmentation)

实例分割要求不仅框出物体，还要分割出物体的精确像素掩膜 (Mask)。

### 6.1 Mask R-CNN
在 Faster R-CNN 的基础上增加了一个用于预测 Mask 的分支。
* **RoI Align**：这是 Mask R-CNN 的核心贡献。
    * **问题**：传统的 RoI Pooling 存在两次量化取整操作（坐标取整），导致特征图与原图区域无法精确对齐。这对分类影响不大，但对像素级分割是致命的。
    * **解决**：RoI Align 取消了取整操作，使用**双线性插值**在浮点坐标上计算特征值，实现了像素级的精确对齐。

---

### 总结 (Summary)

* **两阶段 (Faster R-CNN)**：先生成框再分类，精度高，是很多任务的基准。
* **单阶段 (RetinaNet/YOLO)**：直接预测，速度快，Focal Loss 解决了样本不平衡问题。
* **Transformer (DETR)**：端到端集合预测，去除了 Anchor 和 NMS，通过全局注意力机制处理物体关系。
* **实例分割 (Mask R-CNN)**：利用 RoI Align 解决了特征不对齐问题，实现了检测与分割的统一。