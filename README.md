# SSAU-Net: 基于尺度敏感注意力机制的增强U-Net遥感图像建筑物提取

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

本项目提出了一种改进的**SSAU-Net**模型，通过引入尺度敏感的注意力机制，显著提升了高分辨率遥感图像中建筑物的提取精度。代码基于PyTorch实现，在INRIA数据集上达到SOTA性能。

## 模型亮点 ✨
- **尺度敏感注意力机制**：针对不同层次特征动态调整空间/通道注意力的组合顺序
- **双路径特征增强**：
  - 高层特征优先通道注意力 → 强化语义信息
  - 低层特征优先空间注意力 → 保留几何细节
- **轻量化设计**：通过7×7卷积和共享MLP实现高效注意力计算
- **多尺度特征融合**：在编码器中嵌入注意力模块，增强特征表达能力

## 网络架构
![SSAU-Net Architecture](./images/SSAU-Net Architecture.png)  

## 注意力机制

### 通道注意力机制
```bash
 class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
```
### 空间注意力模块
```bash
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
```

## 实验结果

### 消融实验
| 模型变体       | 注意力组合方式       | F1 Score (%) |
|----------------|---------------------|-------------|
| U-Net          | 无注意力            | 85.72       |
| CSAU-Net       | 通道→空间           | 85.89       |
| SCAU-Net       | 空间→通道           | 85.85       |
| SSAU-Net       | 尺度敏感组合        | **86.07**   |

## 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+

### 安装依赖
```bash
pip install -r requirements.txt
