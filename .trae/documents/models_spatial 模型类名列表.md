根据 `models/spatial/__init__.py` 的注册信息，以下是 `models/spatial` 目录下可用的模型类名列表（按类别整理）：

### 1. CNN 模型 (Convolutional Networks)
- **UNet**: 经典 U-Net 架构
- **UNetPlusPlus**: 嵌套 U-Net (UNet++)
- **FNO2d**: Fourier Neural Operator (2D)
- **UFNOUNet**: U-Net 结合 FNO 瓶颈层

### 2. Transformer 模型 (Transformer-based)
- **SegFormer**: SegFormer 语义分割架构
- **UNetFormer**: U-Net 结合 Transformer
- **SegFormerUNetFormer**: SegFormer 与 UNetFormer 的混合体

### 3. MLP 模型 (Multi-Layer Perceptrons)
- **MLPMixer**: MLP-Mixer 架构
- **LIIFModel**: Learning Implicit Image Function (隐式神经表示)
- **MLPModel** (别名 **MLP**): 标准多层感知机
- **CoordinateEncoder**: 坐标编码器 (辅助模块)

### 4. 混合模型 (Hybrid Architectures)
- **HybridModel**: Attention + FNO + UNet 多分支混合
- **SwinUNet**: Swin Transformer 结合 U-Net

### 5. 基础 Transformer 模型 (Base Transformers)
- **VisionTransformer** (别名 **ViT**): 标准 ViT
- **SwinTransformerTiny** (别名 **SwinT**): Swin Transformer Tiny
- **SwinTWithEncoder**: 带编码器的 Swin Transformer
- **Transformer**: 标准 Transformer (Encoder-Decoder)

### 6. 轻量级/稀疏/SR 模型 (Lightweight/Sparse/Super-Resolution)
- **ConvUNetLite**: 轻量级卷积 U-Net
- **ResNetLite**: 轻量级 ResNet
- **CNNAttnLite**: CNN + Channel Attention (轻量级)
- **ConvGateLite**: 门控卷积网络 (轻量级)
- **SparseAttentionEncoder**: 稀疏注意力编码器
- **SparseSwinUNet**: 稀疏 Swin U-Net

### 文件目录结构 (部分关键文件)
```text
models/spatial/
├── unet.py
├── unet_plus_plus.py
├── fno2d.py
├── ufno_unet_bottleneck.py
├── segformer.py
├── unetformer.py
├── segformer_unetformer.py
├── mlp.py                  (含 MLPModel)
├── coordinate_encoder.py   (含 CoordinateEncoder)
├── mlp_mixer.py
├── liif.py
├── hybrid.py
├── swin_unet.py
├── vit.py
├── swin_t.py
├── conv_unet_lite.py
├── resnet_lite.py
├── cnn_attn_lite.py
├── conv_gate_lite.py
└── ...
```
您可以直接在代码中通过 `from models.spatial import <ModelName>` 进行导入。