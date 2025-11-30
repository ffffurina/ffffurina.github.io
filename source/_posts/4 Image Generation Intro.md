---
categories:
- 学习笔记
date: '2025-11-27 14:35:16'
title: 4 Image Generation Intro
---
本讲涵盖了深度生成模型的四大主流范式：自回归模型 (Autoregressive Models)、变分自编码器 (VAE)、生成对抗网络 (GAN) 以及扩散模型 (Diffusion Models)。
可以参考深度学习高级话题[]

---

# Lecture 21: Image Generation (图像生成)

生成式学习的核心目标是学习数据的潜在分布 $P(x)$，从而能够生成与真实数据分布相似的新样本。

## 1. 自回归模型 (Autoregressive Models)

### 1.1 核心思想
自回归模型将联合概率分布 $p(x)$ 分解为一系列条件概率的乘积。假设图像 $x$ 由序列 $(x_1, x_2, ..., x_T)$ 组成（例如按光栅扫描顺序的像素），则：

$$
p(x) = \prod_{t=1}^{T} p(x_t | x_1, ..., x_{t-1})
$$

### 1.2 训练与生成
* **训练**：通过最大似然估计 (MLE) 优化参数 $W$，F即最大化 $\sum_i \log p(x^{(i)})$. 
* **生成**：是一个串行过程。先生成 $x_1$，再根据 $x_1$ 生成 $x_2$，依此类推。

### 1.3 经典模型
* **PixelRNN / PixelCNN**：逐像素生成图像。利用 RNN (LSTM) 或 掩膜卷积 (Masked CNN) 来建模长程依赖。输出层通常使用 Softmax 在 $[0, 255]$ 的离散空间上预测像素值。 
* **ImageGPT**：基于 Transformer 的自回归模型。将图像像素序列化后输入 GPT，利用注意力机制捕捉全局依赖。 

---

## 2. 变分自编码器 (Variational Autoencoder, VAE)

### 2.1 自编码器的局限
普通的自编码器 (Autoencoder) 旨在通过 Encoder压缩数据 ($z=e(x)$) 并通过 Decoder 重建数据 ($\hat{x}=d(z)$) 。但它不是生成模型，因为其潜空间 (Latent Space) 通常是不连续且无规则的，随机采样 $z$ 无法生成有意义的图像。

### 2.2 VAE 的改进
VAE 强制潜空间服从特定分布（通常是标准正态分布 $\mathcal{N}(0, I)$）。
* **Encoder**：预测分布的参数（均值 $\mu_x$ 和方差 $\sigma_x$），而不是直接预测 $z$。
* **重参数化技巧 (Reparameterization Trick)**：为了使采样过程可导，引入噪声 $\epsilon \sim \mathcal{N}(0, I)$，令：

$$
z = \mu_x + \sigma_x \cdot \epsilon
$$

这样梯度可以反向传播回 $\mu_x$ 和 $\sigma_x$ 。

### 2.3 损失函数
VAE 的损失函数由**重建损失**和**KL 散度**组成（推导自 ELBO）：

$$
\mathcal{L} = \|\mathbf{x} - d(z)\|^2 + \lambda \cdot D_{KL}\big(\mathcal{N}(\mu_x, \sigma_x) \| \mathcal{N}(0, I)\big)
$$

其中 KL 散度项迫使潜变量分布逼近标准正态分布，防止模型通过过拟合均值和极小方差来退化成普通 AE。

---

## 3. 生成对抗网络 (Generative Adversarial Network, GAN)

### 3.1 博弈论框架
GAN 由两个网络组成，进行极小极大 (Minimax) 博弈 ：
* **生成器 (G)**：从随机噪声 $z \sim p_z(z)$ 映射到数据空间，试图生成逼真的样本 $G(z)$ 来欺骗判别器。
* **判别器 (D)**：试图区分真实样本 $x \sim p_{data}(x)$ 和生成样本 $G(z)$。
### 3.2 目标函数
GAN 的优化目标是：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

* **训练 D**：最大化分辨能力。
* **训练 G**：最小化 $\log(1 - D(G(z)))$，或者在实践中通常最大化 $\log D(G(z))$ 以缓解梯度消失问题。

### 3.3 典型架构：DCGAN
深度卷积 GAN (Deep Convolutional GAN) 确立了 GAN 在图像生成中的标准架构，例如使用转置卷积 (Transposed Convolution / Deconvolution) 进行上采样，去除全连接层等 。

### 3.4 StyleGAN
NVIDIA 提出的 StyleGAN 引入了 Mapping Network 将噪声 $z$ 映射为解耦的 $w$ 空间，并使用 AdaIN (Adaptive Instance Normalization) 将风格信息注入到生成过程的各个层级，实现了极高质量和可控的人脸生成。

---

## 4. 扩散模型 (Diffusion Models)

这是目前最前沿的生成模型，也是本讲的重点。

### 4.1 基本原理
扩散模型包含两个过程：
1.  **前向过程 (Forward Process)**：逐步向数据添加高斯噪声，直到变成纯噪声。这是一个马尔可夫链。
2.  **逆向过程 (Inverse Process)**：训练一个神经网络去逐步去除噪声，从纯噪声中恢复出数据。

### 4.2 前向过程 (加噪)
给定真实图像 $x_0$，逐步添加高斯噪声得到 $x_1, ..., x_T$。条件概率定义为：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I})
$$

其中 $\alpha_t$ 是预定义的方差调度参数。
利用高斯分布的性质，可以直接从 $x_0$ 采样得到任意时刻 $t$ 的 $x_t$ ：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。这意味着 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。

### 4.3 逆向过程 (去噪)
我们要学习逆向转移分布 $p_\theta(x_{t-1} | x_t)$。当步长足够小时，逆向过程也可以近似为高斯分布：

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**训练目标**：实际上是训练一个噪声预测网络 $\epsilon_\theta(x_t, t)$ 来预测加入的噪声 $\epsilon$。
简化后的损失函数为 MSE：

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$

### 4.4 采样过程
从标准正态分布 $x_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，迭代去噪：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

其中 $z \sim \mathcal{N}(0, \mathbf{I})$ 是随机噪声项（当 $t>1$ 时加入）。

### 4.5 潜在扩散模型 (Latent Diffusion Models / Stable Diffusion)
为了降低计算成本，LDM 将扩散过程移至预训练的自动编码器 (VQ-GAN 或 KL-AE) 的**潜空间 (Latent Space)** 中进行。
* **流程**：Pixel Space $\xrightarrow{\text{Encoder}}$ Latent Space $\xrightarrow{\text{Diffusion}}$ Latent Space $\xrightarrow{\text{Decoder}}$ Pixel Space。
* **条件生成**：通过 Cross-Attention 机制引入文本 (CLIP text encoder) 或语义图作为条件控制生成。

### 4.6 总结对比
* **GAN**：采样快，但训练不稳定 (Mode Collapse)，难以覆盖整个分布。
* **VAE**：训练稳定，能覆盖分布，但生成的图像通常较模糊。
* **Diffusion**：生成质量极高，训练稳定，能覆盖分布，主要缺点是采样速度慢（需要多步迭代）。