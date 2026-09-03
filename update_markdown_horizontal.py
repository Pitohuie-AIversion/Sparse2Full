import re

files = [
    'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md',
    'thesis_paper/manuscript_5_chapter/thesis_full.md'
]

old_text = "从上述数据可以观察到，基于 EDSR 的统一架构在该数据集上表现出了极低的相对 L2 误差（0.0331）和极高的峰值信噪比（38.79 dB）。对应的多个测试样本的可视化预测结果（从上至下依次为：样本 1、样本 2 和样本 9192）如下图所示：\n\n![Darcy Flow 验证多样本可视化](../../paper_package/figs/DarcyFlow/darcy_flow_verification_combined.png)"

new_text = "从上述数据可以观察到，基于 EDSR 的统一架构在该数据集上表现出了极低的相对 L2 误差（0.0331）和极高的峰值信噪比（38.79 dB）。对应的多个测试样本的可视化预测结果（从左至右依次为：样本 2、样本 1 和样本 9192）如下图所示：\n\n![Darcy Flow 验证多样本可视化](../../paper_package/figs/DarcyFlow/darcy_flow_verification_horizontal.png)"

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path}")
    else:
        print(f"Text not found in {file_path}, maybe already updated or slightly different.")

