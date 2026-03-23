#!/usr/bin/env python3
"""
检查时序NAR训练结果
"""

import json
from pathlib import Path

def check_training_results():
    """检查训练结果"""
    
    print("=" * 60)
    print("🚀 时序NAR模型训练结果检查")
    print("=" * 60)
    
    # 检查训练历史
    history_file = Path("runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/training_history.json")
    
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        train_losses = history['train_losses']
        best_val_loss = history['best_val_loss']
        
        print(f"\n📊 训练完成情况:")
        print(f"   ✅ 训练已完成 {len(train_losses)} 轮")
        print(f"   📉 最终训练损失: {train_losses[-1]:.6f}")
        print(f"   🎯 最佳验证损失: {best_val_loss:.6f}")
        print(f"   📈 最小训练损失: {min(train_losses):.6f}")
        print(f"   📊 平均训练损失: {sum(train_losses)/len(train_losses):.6f}")
        
        # 检查收敛性
        last_10 = train_losses[-10:] if len(train_losses) >= 10 else train_losses
        convergence_std = sum([(x - sum(last_10)/len(last_10))**2 for x in last_10]) / len(last_10)
        convergence_std = convergence_std ** 0.5
        
        print(f"   🔄 最后10轮标准差: {convergence_std:.6f}")
        print(f"   ✅ 收敛状态: {'良好' if convergence_std < 0.01 else '需要更多轮次'}")
        
    else:
        print("❌ 未找到训练历史文件，训练可能未完成或失败")
    
    # 检查输出文件
    output_dir = Path("runs/temporal_nar_100epochs")
    
    print(f"\n📁 输出文件检查:")
    print(f"   配置文件: {'✅' if (output_dir / 'config_merged.yaml').exists() else '❌'}")
    print(f"   训练日志: {'✅' if (output_dir / 'logs' / 'train.log').exists() else '❌'}")
    print(f"   检查点目录: {'✅' if (output_dir / 'checkpoints').exists() else '❌'}")
    print(f"   可视化目录: {'✅' if (output_dir / 'visualizations').exists() else '❌'}")
    
    # 检查模型检查点
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt")) + list(checkpoints_dir.glob("*.pth"))
        print(f"   模型检查点: {len(ckpt_files)} 个文件")
        for ckpt in ckpt_files:
            print(f"     - {ckpt.name}")
    else:
        print("   模型检查点: ❌ 未找到")
    
    # 检查日志文件大小
    log_file = output_dir / "logs" / "train.log"
    if log_file.exists():
        log_size = log_file.stat().st_size / 1024  # KB
        print(f"   日志文件大小: {log_size:.1f} KB")
    
    print("\n" + "=" * 60)
    
    # 返回训练是否成功
    return history_file.exists() and len(train_losses) > 0 if history_file.exists() else False

if __name__ == "__main__":
    success = check_training_results()
    if success:
        print("🎉 训练成功完成！")
    else:
        print("❌ 训练未完成或失败")