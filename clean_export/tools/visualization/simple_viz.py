import json
from pathlib import Path

# 设置路径
base_dir = Path(".")
data_path = base_dir / "runs/temporal_nar_100epochs/TemporalNAR-DR2D-128-100epochs-s2025/training_history.json"
save_dir = base_dir / "runs/temporal_nar_100epochs/visualizations"

# 创建保存目录
save_dir.mkdir(parents=True, exist_ok=True)

print("🚀 开始生成时序NAR训练可视化报告...")

# 加载数据
print("📊 正在加载训练数据...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

train_losses = data['train_losses']
val_losses = data.get('val_losses', [])
best_val_loss = data.get('best_val_loss', min(val_losses) if val_losses else None)

print(f"✅ 成功加载 {len(train_losses)} 轮训练数据")

# 计算基本统计
total_epochs = len(train_losses)
final_train_loss = train_losses[-1]
best_train_loss = min(train_losses)
train_loss_reduction = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100

# 生成报告
report_content = f"""时序NAR模型训练可视化报告
{'='*60}

📊 训练配置和结果摘要
{'='*60}

🎯 基本信息
• 总训练轮次: {total_epochs}
• 最终训练损失: {final_train_loss:.6f}
• 最佳训练损失: {best_train_loss:.6f}
• 训练损失降幅: {train_loss_reduction:.2f}%

📈 验证统计
• 最佳验证损失: {best_val_loss:.6f if best_val_loss else 'N/A'}

📊 详细训练数据
训练损失序列 (前10轮): {train_losses[:10]}
训练损失序列 (后10轮): {train_losses[-10:]}
"""

if val_losses:
    val_loss_reduction = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
    report_content += f"""
验证损失降幅: {val_loss_reduction:.2f}%
验证损失序列 (前10轮): {val_losses[:10]}
验证损失序列 (后10轮): {val_losses[-10:]}
"""

# 保存报告
report_path = save_dir / "training_visualization_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 训练可视化报告已保存: {report_path}")

# 生成CSV格式的数据
csv_content = "epoch,train_loss"
if val_losses:
    csv_content += ",val_loss"
csv_content += "\n"

for i, loss in enumerate(train_losses):
    csv_content += f"{i+1},{loss}"
    if val_losses and i < len(val_losses):
        csv_content += f",{val_losses[i]}"
    csv_content += "\n"

csv_path = save_dir / "training_data.csv"
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write(csv_content)

print(f"✅ 训练数据CSV已保存: {csv_path}")
print(f"🎉 可视化报告生成完成！")
print(f"📁 保存位置: {save_dir}")