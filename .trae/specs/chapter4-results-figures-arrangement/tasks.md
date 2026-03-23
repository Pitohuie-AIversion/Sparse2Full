# Tasks
- [x] Task 1: 确定图像来源与样本：检索或请用户提供 `ob`, `prd`, `gt`, `err` 图像所在的具体路径，并挑选出要展示的典型测试样本。
- [x] Task 2: 编写/更新图像拼接脚本：编写 Python 脚本将选定的 `ob`, `prd`, `gt`, `err` 拼接为符合学术规范的组合对比图（如 fig4-1_vis_results 等）。
- [x] Task 3: 生成结果组图：运行脚本生成新的对比图并保存至 `thesis_paper/manuscript_5_chapter/images/` 目录。
- [x] Task 4: 更新第四章文档：在 `chapter4_results_verification.md` 中引用新生成的图像，检查并调整图注、图号和正文引用文本。
- [x] Task 5: 编译验证：运行 `tools/convert_thesis.py` 生成最终文档，检查图片排版和清晰度。

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4