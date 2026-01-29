I have identified the root cause of the indentation issues:
1.  **Lists**: The thesis template's `List Paragraph` style inherits from `Normal`, which carries a 2-character indentation (approx 0.33"). This causes list items to be double-indented or misaligned.
2.  **Tables**: Table cells were falling back to `Normal` (or `正文1`), inheriting the same unwanted indentation.
3.  **Captions**: Image and table captions need to be explicitly mapped to `图名中文` to avoid inheriting the body text indentation.

### Solution Plan

I will fix this by creating a **Modified Template** and an **Enhanced Style Filter**.

#### 1. Fix the Template (`tools/modify_template.py`)
I will write a Python script to programmatically modify the `大连海事大学硕士学位论文格式模版20181122.docx` template:
*   **Fix `List Paragraph`**: Force its `first_line_indent` to 0 to prevent double indentation.
*   **Create `Table Text`**: Add a new style specifically for table contents with 0 indentation (inheriting font from `Normal` but removing the indent).

#### 2. Enhance Style Logic (`tools/thesis_style.lua`)
I will update the Lua filter to be context-aware:
*   **Lists**: Force all paragraphs inside lists to use the fixed `List Paragraph` style.
*   **Tables**: Force all paragraphs inside table cells to use the new `Table Text` style.
*   **Captions**: Explicitly map Table/Figure captions to `图名中文`.
*   **Body Text**: Continue mapping standard paragraphs to `正文1` (preserving the correct 2-char indent for main text).

#### 3. Execution
*   Generate `template_fixed.docx`.
*   Re-run the conversion using the new template and filter.
*   Verify the output.

This ensures that **Body Text** has the required indentation, while **Lists, Tables, and Captions** are correctly aligned without unwanted indentation.
