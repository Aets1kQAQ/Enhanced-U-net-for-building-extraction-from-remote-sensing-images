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

## 编码器-解码器结构
编码器：通过4级下采样（encoder模块）逐步提取高层语义特征，每级包含卷积块（x2conv）和池化（MaxPool2d）

解码器：通过4级上采样（decoder模块）恢复空间分辨率，每级包含转置卷积（ConvTranspose2d）和跳跃连接（Skip Connection）
### 编码器部分 (Contracting Path)
```
self.start_conv = x2conv(in_channels, 64)          # 初始卷积块
self.down1 = encoder(64, 128)                      # 下采样阶段1
self.down2 = encoder(128, 256)                     # 下采样阶段2
self.down3 = encoder(256, 512)                     # 下采样阶段3
self.down4 = encoder(512, 1024)                    # 下采样阶段4
self.middle_conv = x2conv(1024, 1024)              # 中间过渡卷积
```
### 解码器部分 (Expansive Path)
```
self.up1 = decoder(1024, 512)                      # 上采样阶段1
self.up2 = decoder(512, 256)                       # 上采样阶段2
self.up3 = decoder(256, 128)                       # 上采样阶段3
self.up4 = decoder(128, 64)                        # 上采样阶段4
self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 最终分类卷积
```
## 注意力机制

### 通道注意力机制
```
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
```
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
## 注意力机制实现
#### 1. encoder模块（无注意力机制）
```
class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)  # 特征提取
        x = self.pool(x)       # 下采样
        return x
```
#### 2. encoder1模块（先空间后通道）
```
class encoder1(nn.Module):
    def __init__(self, in_channels, out_channels):
        ...
        self.spatial_attention = SpatialAttention()  # 先空间
        self.channel_attention = ChannelAttention()  # 后通道
```
#### 3. encoder2模块（先通道后空间）
```
class encoder2(nn.Module):
    def __init__(self, in_channels, out_channels):
        ...
        self.channel_attention = ChannelAttention()  # 先通道
        self.spatial_attention = SpatialAttention()  # 后空间
```

#### 1. decoder模块（无注意力机制）
```
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)  # 转置卷积上采样
        self.up_conv = x2conv(in_channels, out_channels)  # 双层卷积块

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)  # 上采样操作
        
        # 尺寸对齐处理
        if x.shape[2:] != x_copy.shape[2:]:
            if interpolate:
                x = F.interpolate(x, size=x_copy.shape[2:], mode="bilinear")  # 插值对齐
            else:
                x = F.pad(x, [...] )  # 填充对齐
        
        x = torch.cat([x_copy, x], dim=1)  # 跳跃连接
        x = self.up_conv(x)  # 卷积融合
        return x
```
#### 2. decoder1模块（先空间后通道）
```
class decoder2(decoder):
    def __init__(self, in_channels, out_channels):
        ...
        self.spatial_attention = SpatialAttention()             # 空间注意力
        self.channel_attention = ChannelAttention(in_channels)  # 通道注意力
    def forward(...):
        ...
        x = torch.cat([x_copy, x], dim=1)
        x = spatial_att(x) * x  # 先应用空间注意力
        x = channel_att(x) * x  # 再应用通道注意力
        x = self.up_conv(x)
```
#### 3. decoder2模块（先通道后空间）
```
class decoder1(decoder):
    def __init__(self, in_channels, out_channels):
        ...
        self.channel_attention = ChannelAttention(in_channels)  # 通道注意力
        self.spatial_attention = SpatialAttention()             # 空间注意力
    def forward(...):
        ...
        x = torch.cat([x_copy, x], dim=1)
        x = channel_att(x) * x  # 先应用通道注意力
        x = spatial_att(x) * x  # 再应用空间注意力
        x = self.up_conv(x)
```
## 实验过程
### U-Net（无注意力）=======》A
```
self.down1 = encoder(64, 128)                      # 下采样阶段1
self.down2 = encoder(128, 256)                     # 下采样阶段2
self.down3 = encoder(256, 512)                     # 下采样阶段3
self.down4 = encoder(512, 1024)                    # 下采样阶段4
```
### CSAU-Net（先通道后空间）=======》B
```
self.down1 = encoder1(64, 128)                      # 下采样阶段1
self.down2 = encoder1(128, 256)                     # 下采样阶段2
self.down3 = encoder1(256, 512)                     # 下采样阶段3
self.down4 = encoder1(512, 1024)                    # 下采样阶段4
```
### SCAU-Net（先空间后通道）=======》C
```
self.down1 = encoder2(64, 128)                      # 下采样阶段1
self.down2 = encoder2(128, 256)                     # 下采样阶段2
self.down3 = encoder2(256, 512)                     # 下采样阶段3
self.down4 = encoder2(512, 1024)                    # 下采样阶段4
```
### SSAU-Net（尺度敏感组合）=======》D
```
self.down1 = encoder2(64, 128)                      # 下采样阶段1
self.down2 = encoder2(128, 256)                     # 下采样阶段2
self.down3 = encoder1(256, 512)                     # 下采样阶段3
self.down4 = encoder1(512, 1024)                    # 下采样阶段4
```

## 实验结果

### 消融实验（下采样）
| 模型变体       | 注意力组合方式 | F1 Score (%) | 代码实现方式 |
|----------------|---------|-------------|--------|
| U-Net          | 无注意力    | 85.72       | A      |
| CSAU-Net       | 先通道→空间  | 85.89       | B      |
| SCAU-Net       | 先空间→通道  | 85.85       | C      |
| SSAU-Net       | 尺度敏感组合  | **86.07**   | D      |

## 实验开始

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+

### 安装依赖
```bash
pip install -r requirements.txt
```

### 前期工作