# LaTeX 写作工程（XeLaTeX + biblatex(APA) + biber）

## 依赖

- Windows + TeX Live 2023（需包含：xelatex、latexmk、biber）
- VS Code + LaTeX Workshop 扩展

在 PowerShell 中确认：

```powershell
xelatex --version
latexmk --version
biber --version
```

## 工程约定

- 主文件：`main.tex`
- 章节拆分：`chapters/*.tex`，使用 `\input{...}`（不使用 `\include`，避免 outDir 子目录 aux 问题）
- 参考文献：`references.bib`，使用 `biblatex + biber`（不使用 BibTeX）
- 输出目录：`out/`

## 一键编译

在 `manuscript/` 目录运行：

```powershell
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=out -e '$bibtex_use = 2;' main.tex
```

预期输出：`out/main.pdf`

## Clean

```powershell
latexmk -C -outdir=out main.tex
```

如果曾经在工程根目录生成过 `main.aux/main.bcf/main.run.xml/main.bbl` 等旧产物，建议手动删除它们，保证所有中间文件只出现在 `out/`。

## VS Code（LaTeX Workshop）

本工程已提供 `.vscode/settings.json`：

- 保存自动编译（onSave）
- 编译链路：`latexmk (xelatex + biber)`
- 输出到 `out/`

打开 `main.tex` 后，保存任意 `.tex` 或 `references.bib`，应自动触发构建并刷新引用与参考文献。

## 常见故障排查

### 1) 引用显示为 “??” / 参考文献不更新

- 确认实际执行的是 `biber`（不是 `bibtex`）：latexmk 需要启用 biber 模式
- 首次引入新条目时，通常需要完整跑通一次链路：XeLaTeX → biber → XeLaTeX（latexmk 会自动完成）
- 检查 `out/` 中是否生成了 `main.bcf` 与 `main.bbl`

### 2) Package biblatex: File 'main.bbl' not created by biblatex

常见根因是旧的 `main.bbl`（由 BibTeX 生成）或 biber 未运行导致链路断裂。

处理方式：

1. 清理旧产物（推荐直接删除 `out/` 中间文件或执行 clean）
2. 重新编译：

```powershell
latexmk -C -outdir=out main.tex
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -outdir=out main.tex
```

### 3) outDir 导致 aux/中间文件路径混乱

- 不要使用 `\include`（会尝试写子目录 aux）
- 统一使用 `\input{chapters/xxx.tex}`，并确保编译始终带 `-outdir=out`

### 4) 中文/ctex 字体问题

- TeX Live 安装不完整或系统字体缺失时可能报错
- 优先确认 TeX Live 安装完整，并重启 VS Code/终端让 PATH 生效
