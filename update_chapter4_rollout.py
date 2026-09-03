import os

filepath = 'thesis_paper/manuscript_5_chapter/chapter4_results_verification.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

old_text = "![图 4-4: 时空预测任务中的误差累积（Rollout Error）分析。随着预测步长（Time Step）的增加，Ours (Seq-EDSR) 的累积误差增长最为缓慢，表现出优异的长时稳定性；而 UNet 与 FNO 则出现了较快的误差漂移。](images/fig4-7_rollout_error.png)"

new_text = "![图 4-4: 时空预测任务中的误差累积（Rollout Error）分析。随着预测步长（Time Step）的增加，Ours (Seq-EDSR) 的累积误差增长最为缓慢，表现出优异的长时稳定性；而 UNet 与 FNO 则出现了较快的误差漂移。](../../figures/rollout/fig4-4_rollout_error.png)"

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Rollout image link updated successfully!")
else:
    print("Old text not found!")

