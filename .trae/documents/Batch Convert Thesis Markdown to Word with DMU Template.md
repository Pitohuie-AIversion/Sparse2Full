I have analyzed the Word template (`大连海事大学硕士学位论文格式模版20181122.docx`) and your Markdown files.

**Analysis Results:**
1.  **Template Styles**: The template contains standard styles (`Heading 1`, `Heading 2`, `Normal`) as well as custom Chinese styles (`摘要`, `正文1`, `图名中文`).
2.  **Markdown Format**: Your Markdown files use standard headers (`#`, `##`, `###`), which automatically map to `Heading 1`, `Heading 2`, `Heading 3` in Word.

**Plan:**

I will create a Python script `tools/batch_convert_thesis.py` to automate the conversion using `pandoc`.

**1. Conversion Logic:**
   - Use `pandoc` with the `--reference-doc` argument. This forces the output Word documents to inherit all fonts, margins, and styles from your DMU thesis template.
   - **Mapping**:
     - `# Title` → `Heading 1` (matches your template's chapter style)
     - `## Section` → `Heading 2`
     - `text` → `Normal` (or template's default body text)

**2. Execution Steps:**
   - Ensure the output directory `thesis_paper/manuscript_gpt_review/pandoc_word/` exists.
   - Loop through all the Markdown files you provided (`chapter0_abstract.md` to `writing_checklist.md`).
   - Generate corresponding `.docx` files in the output directory.

This approach ensures "one-click" transcoding while strictly respecting the university's format template.
