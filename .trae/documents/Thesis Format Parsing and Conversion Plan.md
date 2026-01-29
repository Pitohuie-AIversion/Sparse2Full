Based on the analysis of the template (`大连海事大学硕士学位论文格式模版20181122.docx`), I have identified the correct style mapping for your thesis conversion.

### Style Mapping Strategy
I will map Markdown elements to the following Word styles found in your template:

| Markdown Element | Word Style Name in Template | Notes |
| :--- | :--- | :--- |
| **Chapter Title** (`# Title`) | `样式 标题 1 + 段后: 1 行` | *Note: The template uses this long name for "Chapter 1".* |
| **Section Title** (`## Title`) | `Heading 2` | Standard |
| **Subsection** (`### Title`) | `Heading 3` | Standard |
| **Body Text** | `正文1` | *Note: The template explicitly mentions `正文` style for body text.* |
| **Figure Captions** | `图名中文` / `图名英文` | Will map default image captions to `图名中文` |
| **Table Captions** | `图名中文` | (Template uses same style for Table titles in examples) |
| **References** | `参考文献正文` | For the bibliography section |
| **Abstract Title** | `摘要题目` | For `# 摘要` |
| **Abstract Body** | `摘要` | For abstract text |

### Implementation Plan
I will create a toolchain to automate this conversion:

1.  **Lua Filter (`thesis_style.lua`)**: A script for Pandoc that intercepts document elements and assigns the specific custom styles defined above (especially the complex ones like `样式 标题 1...`).
2.  **Conversion Script (`convert_thesis.py`)**: A Python script that:
    - Takes your list of Markdown files.
    - Runs `pandoc` with the template and the Lua filter.
    - Outputs individual `.docx` files for each chapter to `docx_output/`.
    - (Optional) Can merge them if needed, but separate files are safer for final editing.

### Execution
I will generate these scripts and run the conversion for all the files you listed.

**Output Location**: `thesis_paper/manuscript_gpt_review/docx_output/`
