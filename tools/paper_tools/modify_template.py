import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Source template (read-only)
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/DMUpapertemplate/大连海事大学硕士学位论文格式模版20181122.docx")
# Target template (to be used by pandoc)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/DMUpapertemplate/template_fixed.docx")

def modify_template():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        return

    doc = docx.Document(TEMPLATE_PATH)
    print(f"Loaded template: {TEMPLATE_PATH}")

    # Helper to safe-get or create paragraph style
    def ensure_style(name, base_name='Normal'):
        if name in doc.styles:
            print(f"Modifying existing style: {name}")
            return doc.styles[name]
        else:
            print(f"Creating new style: {name}")
            style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            if base_name in doc.styles:
                style.base_style = doc.styles[base_name]
            return style

    # 1. Fix 'List Paragraph' (Remove First Line Indent)
    # Even if we stop mapping in Lua, Pandoc might still use this by default for some lists.
    lp = ensure_style('List Paragraph')
    lp.paragraph_format.first_line_indent = 0
    lp.paragraph_format.left_indent = Pt(0) # Reset left indent to let numbering handle it

    # 2. Create/Fix 'Table Text' style (No Indent)
    tt = ensure_style('Table Text')
    tt.paragraph_format.first_line_indent = 0
    tt.paragraph_format.left_indent = 0

    # 3. Create 'Formula' style (Compact spacing for equations)
    # This solves the "newline gap" issue.
    # We set spacing_after to 0 to keep text tight.
    # We also ensure no first-line indent, so centered equations are truly centered.
    formula = ensure_style('Formula')
    formula.paragraph_format.first_line_indent = 0
    formula.paragraph_format.space_before = Pt(0)
    formula.paragraph_format.space_after = Pt(0)
    formula.paragraph_format.line_spacing = 1 # Single spacing
    # Optional: Center align formula if desired, but Pandoc usually handles OMML alignment internally.
    # Let's just ensure the container paragraph doesn't mess it up.

    # 4. Create/Fix '正文1' (Main Body Text)
    # Ensure it has proper indentation and doesn't break flow
    zw1 = ensure_style('正文1', base_name='Normal')
    zw1.paragraph_format.first_line_indent = Pt(24) # Approx 2 chars for 12pt font (standard thesis font size)
    zw1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # Ensure no crazy spacing to keep equations tight
    zw1.paragraph_format.space_before = Pt(0)
    zw1.paragraph_format.space_after = Pt(0)
    
    # 5. Fix 'Compact List' (Optional, for manual tight lists if needed)
    cl = ensure_style('Compact List')
    cl.paragraph_format.first_line_indent = 0
    cl.paragraph_format.space_after = Pt(0)
    cl.paragraph_format.line_spacing = 1

    # Save
    doc.save(OUTPUT_PATH)
    print(f"Saved fixed template to: {OUTPUT_PATH}")

if __name__ == "__main__":
    modify_template()
