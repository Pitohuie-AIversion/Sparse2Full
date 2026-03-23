# 修复 DOCX 导出图片缺失的计划

## 1. 问题分析
在 `appendix.md` 导出为 DOCX 时，模型结构图（SVG）未能成功嵌入。经过排查，发现原因在于 `tools/convert_thesis.py` 脚本在调用 Pandoc 转换为 DOCX 时，没有指定 `--resource-path` 参数。

由于该脚本是在项目根目录下执行的，Pandoc 会默认将相对路径（如 `../figures_nn/build_export_j2/...`）相对于项目根目录进行解析，导致无法找到目标文件（实际上这些文件是相对于 Markdown 文件所在的 `thesis_paper/manuscript_5_chapter/` 目录的）。

对于 PDF 生成，由于先生成了 HTML 文件（且我已将其路径修正为与 Markdown 同级），渲染引擎 Prince 能够根据 HTML 的路径正确解析图片相对位置，所以 PDF 中图片正常显示。而 DOCX 是由 Pandoc 直接解析并打包嵌入的，必须通过参数显式告诉 Pandoc 正确的工作路径。

## 2. 执行步骤
1. **修改导出脚本 (`tools/convert_thesis.py`)**：
   - 在 `convert_md_to_docx` 函数的 Pandoc 构建命令 `cmd` 中，增加 `"--resource-path=.:" + os.path.dirname(md_file)` 参数。这会让 Pandoc 在解析相对路径时，优先从 Markdown 文件所在的目录开始查找。
   - （可选优化）将 `convert_md_to_pdf` 函数中的 `--resource-path=.:" + PROJECT_ROOT` 也统一修正为 `os.path.dirname(md_file)`，保持逻辑一致性。
2. **重新导出文档**：
   - 运行 `python tools/convert_thesis.py --format docx` 重新生成所有的 Word 文档。
3. **验证结果**：
   - 检查控制台输出，确认之前出现的 `[WARNING] Could not fetch resource ../figures_nn/...` 警告已消失。
   - 检查 `docx_output/appendix.docx` 文件大小是否增加（说明图片已被打包嵌入）。

如果该计划没问题，请批准执行。