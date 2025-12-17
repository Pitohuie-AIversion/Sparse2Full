"""ARWrapper统一接口使用示例

展示如何使用新的统一接口以及新旧接口的对比。
"""

import torch
import torch.nn as nn
from models.ar import ARWrapper, autoregressive_predict


class SimpleSwinModel(nn.Module):
    """简化的单帧模型示例"""
    def __init__(self, in_ch=4, out_ch=4):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        """统一接口：x [B, C_in, H, W] -> y [B, C_out, H, W]"""
        return self.conv(x)


def demo_unified_interface():
    """演示统一接口的使用"""
    print("=== ARWrapper统一接口演示 ===\n")
    
    # 创建模型
    base_model = SimpleSwinModel(in_ch=4, out_ch=4)
    ar_model = ARWrapper(base_model)
    
    print("✓ 创建ARWrapper模型")
    print(f"  - 基础模型: {type(base_model).__name__}")
    print(f"  - 统一接口: {ar_model._unified_interface}")
    print(f"  - 接口版本: {ar_model.get_model_info()['interface_version']}\n")
    
    # 1. 单帧预测（统一接口）
    print("1. 单帧预测（统一接口）")
    B, C, H, W = 2, 4, 64, 64
    x_single = torch.randn(B, C, H, W)
    
    # 统一接口：forward(x) -> y
    y_single = ar_model(x_single)
    print(f"   输入: x {x_single.shape}")
    print(f"   输出: y {y_single.shape}")
    print("   ✓ 符合统一接口规范 forward(x[B,C_in,H,W]) -> y[B,C_out,H,W]\n")
    
    # 2. 打包输入格式支持
    print("2. 打包输入格式 [baseline, coords, mask]")
    # 模拟打包输入：12通道（4+4+4）
    packed_input = torch.randn(B, 12, H, W)
    y_packed = ar_model(packed_input)
    print(f"   打包输入: {packed_input.shape}")
    print(f"   模型输出: {y_packed.shape}")
    print("   ✓ 自动解包处理\n")
    
    # 3. 时间序列预测（外部函数）
    print("3. 时间序列预测（外部函数）")
    T_in, T_out = 5, 10
    x_seq = torch.randn(B, T_in, C, H, W)
    teacher = torch.randn(B, T_out, C, H, W)
    
    # 使用外部函数进行时间序列预测
    y_seq = autoregressive_predict(
        model=ar_model,
        x_seq=x_seq,
        T_out=T_out,
        teacher=teacher,
        train_mode=True
    )
    
    print(f"   输入序列: {x_seq.shape}")
    print(f"   教师序列: {teacher.shape}")
    print(f"   输出序列: {y_seq.shape}")
    print("   ✓ 时间维度处理外移，保持空间处理统一\n")
    
    # 4. 向后兼容方法
    print("4. 向后兼容方法（推荐迁移）")
    y_seq_compat = ar_model.autoregressive_predict(
        x_seq=x_seq,
        T_out=T_out,
        teacher=teacher,
        train_mode=True
    )
    print(f"   兼容方法输出: {y_seq_compat.shape}")
    print("   ✓ 提供迁移路径\n")
    
    print("=== 接口统一完成 ===")


def demo_old_vs_new_interface():
    """对比新旧接口"""
    print("\n=== 新旧接口对比 ===\n")
    
    base_model = SimpleSwinModel(in_ch=4, out_ch=4)
    ar_model = ARWrapper(base_model)
    
    B, T_in, T_out, C, H, W = 2, 5, 3, 4, 64, 64
    x_seq = torch.randn(B, T_in, C, H, W)
    teacher = torch.randn(B, T_out, C, H, W)
    
    print("旧接口（已废弃）：")
    print("  forward(x_in, T_out, teacher, train_mode) -> (B,T_out,C,H,W)")
    print("  ✗ 参数不一致，违反统一接口")
    print("  ✗ 内部处理时间维度，耦合度高")
    print("  ✗ 不支持输入打包格式\n")
    
    print("新接口（统一规范）：")
    print("  1. forward(x) -> (B,C_out,H,W)")
    print("     ✓ 统一接口：只有x参数")
    print("     ✓ 4D输入输出，符合规范")
    print("     ✓ 支持[baseline, coords, mask]打包")
    print("     ✓ 单帧空间预测，模块化设计\n")
    
    print("  2. autoregressive_predict(model, x_seq, T_out, ...) -> (B,T_out,C,H,W)")
    print("     ✓ 时间维度处理外移")
    print("     ✓ 支持教师强制和scheduled sampling")
    print("     ✓ 保持空间处理统一性")
    print("     ✓ 更好的测试性和可维护性\n")
    
    # 演示新接口的实际使用
    print("新接口使用示例：")
    
    # 单帧预测
    x_single = x_seq[:, -1, :, :, :]  # 取最后一帧
    y_single = ar_model(x_single)
    print(f"  单帧预测: {x_single.shape} -> {y_single.shape}")
    
    # 时间序列预测
    y_seq = autoregressive_predict(
        model=ar_model,
        x_seq=x_seq,
        T_out=T_out,
        teacher=teacher,
        train_mode=True
    )
    print(f"  序列预测: {x_seq.shape} -> {y_seq.shape}")


if __name__ == '__main__':
    # 演示统一接口
    demo_unified_interface()
    
    # 对比新旧接口
    demo_old_vs_new_interface()
    
    print("\n✅ ARWrapper接口统一完成！")
    print("📋 技术文档: .trae/documents/arwrapper_interface_unification.md")
    print("🧪 测试文件: tests/test_arwrapper_unified_interface.py")
    print("📖 使用示例: examples/arwrapper_unified_usage.py")