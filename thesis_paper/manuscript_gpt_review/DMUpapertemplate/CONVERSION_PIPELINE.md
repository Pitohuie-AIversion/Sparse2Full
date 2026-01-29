# 论文格式转换工具链架构文档

本文档描述了将 Markdown 格式的学位论文自动转换为符合大连海事大学格式要求（2018版）的 Word 文档的技术架构与工作流。

## 1. 核心架构 (Architecture)

本工具链采用 **Pandoc** 作为核心转换引擎，结合 **Python** 脚本进行流程编排，并使用 **Lua Filter** 进行细粒度的样式注入。

```mermaid
graph TD
    A[Markdown 源文件] -->|python convert_thesis.py| B(Pandoc 转换引擎)
    
    subgraph "样式控制系统"
        C[thesis_style.lua] -->|注入样式元数据| B
        D[template_fixed.docx] -->|提供 Word 样式定义| B
    end
    
    B --> E[Word 文档 (.docx)]
    
    style A fill:#f9f,stroke:#333
    style E fill:#9f9,stroke:#333
    style B fill:#ff9,stroke:#333
```

## 2. 组件说明

### 2.1 转换主控脚本 (`tools/convert_thesis.py`)
- **功能**：批处理所有章节文件，调用 Pandoc 命令。
- **关键配置**：
  - 启用 `markdown+tex_math_single_backslash` 以支持 LaTeX 公式（`$E=mc^2$`, `\[...\]`）。
  - 指定 `--reference-doc` 为修复后的模板。
  - 挂载 `--lua-filter` 处理样式映射。

### 2.2 样式映射过滤器 (`tools/thesis_style.lua`)
- **功能**：在 Pandoc 解析 AST（抽象语法树）时，拦截特定元素并强制应用 Word 样式。
- **关键逻辑**：
  - **标题映射**：
    - `# 摘要` -> `摘要题目`
    - `# 参考文献` -> `参考文献标题`
    - `# 致谢` -> `致谢`
    - `# 第1章...` -> `样式 标题 1 + 段后: 1 行`
  - **排版修正**：
    - **公式 (Formula)**：自动检测包含独立公式 (`DisplayMath`) 的段落，强制应用 `Formula` 样式（紧凑排版）。
    - **正文混合内容**：通过 `Div(Plain)` 包裹技术修复了行内公式导致的**异常换行**问题。
    - **参考文献列表**：检测 `# 参考文献` 标题后的区域，强制将列表和段落应用 `参考文献正文` 样式（悬挂缩进）。
    - **表格 (Table)**：将单元格内容包裹为 `Table Text` 样式（无首行缩进）。
    - **图表标题**：自动识别图片段落，应用 `图名中文` 样式。

### 2.3 模板修复工具 (`tools/modify_template.py`)
- **背景**：原始学校模板 (`...20181122.doc`) 的 `Normal` 样式带有强制缩进，导致列表和表格继承了错误的缩进。
- **功能**：
  - 生成 `template_fixed.docx`。
  - **新建/修复样式**：
    - `Formula`: 段前/段后间距为 0，单倍行距（解决公式上下空行过大问题）。
    - `正文1`: 明确的首行缩进（约2字符），段前/段后间距为 0。
    - `Table Text`: 基于 Normal 但强制缩进为 0。
    - `List Paragraph`: 强制缩进为 0，左对齐。

## 3. 使用方法

### 3.1 运行全量转换
在项目根目录下执行：

```bash
python3 tools/convert_thesis.py
```

输出文件将生成在 `thesis_paper/manuscript_gpt_review/docx_output/` 目录。

### 3.2 维护与修改
- **修改样式映射**：编辑 `tools/thesis_style.lua`。
- **修改基础模板**：
  1. 编辑 `tools/modify_template.py` 修改样式定义。
  2. 运行 `python3 tools/modify_template.py` 重新生成模板。
  3. 运行转换脚本。

## 4. 已解决的关键问题
1.  **公式渲染与间距**：
    - 启用 LaTeX 数学环境支持，输出为 Word 原生公式 (`OMML`)。
    - 通过专用 `Formula` 样式消除了公式上下的多余空行。
2.  **异常换行修复**：
    - 解决了行内公式（Inline Math）导致文本被切断换行的问题（通过 Lua `Div` 包裹修复）。
3.  **缩进异常**：
    - 列表和表格不再继承正文的“首行缩进 2 字符”。
    - 通过 `Table Text` 和 `List Paragraph` 实现了左对齐。
4.  **参考文献对齐**：
    - 手动编写的参考文献列表现在会自动获得正确的“悬挂缩进”格式。
