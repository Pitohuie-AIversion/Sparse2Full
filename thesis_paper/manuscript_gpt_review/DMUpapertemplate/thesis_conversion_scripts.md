# Thesis Conversion Pipeline Scripts

本文件汇总了用于大连海事大学硕士学位论文 Markdown 转 Word (`.docx`) 的核心脚本。
包含 3 个关键文件：
1.  **`convert_thesis.py`**: 主转换程序，调用 Pandoc 执行批量转换。
2.  **`modify_template.py`**: 模板修正脚本，自动在 Word 模板中创建/修复所需样式（如 Formula, Table Text, 正文1）。
3.  **`thesis_style.lua`**: Pandoc Lua 过滤器，用于在转换过程中动态应用样式（处理公式、图名、参考文献等）。

---

## 1. convert_thesis.py (主程序)
路径: `tools/convert_thesis.py`

```python
import os
import subprocess
import sys
from pathlib import Path

def convert_md_to_docx(md_files, template_path, output_dir, filter_path, bib_file=None):
    """
    Convert a list of Markdown files to Docx using Pandoc and a custom template.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return
    if not os.path.exists(filter_path):
        print(f"Error: Filter not found at {filter_path}")
        return

    success_count = 0
    
    for md_file in md_files:
        if not os.path.exists(md_file):
            print(f"Warning: File not found: {md_file}")
            continue
            
        # Determine output filename
        file_name = os.path.basename(md_file)
        name_no_ext = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{name_no_ext}.docx")
        
        print(f"Converting {file_name} -> {output_path}...")
        
        # Construct Pandoc command
        # --reference-doc: Uses styles from the template
        # --lua-filter: Applies custom style mapping
        # --standalone: Produces a standalone document
        cmd = [
            "pandoc",
            md_file,
            "-f", "markdown+tex_math_single_backslash",
            "-o", output_path,
            "--reference-doc", template_path,
            "--lua-filter", filter_path,
            "--standalone"  # Important for applying reference doc styles
        ]

        # Add bibliography support if file exists
        if bib_file and os.path.exists(bib_file):
            cmd.extend([
                "--bibliography", bib_file,
                "--citeproc" # Enable citation processing
            ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  [OK] Converted {file_name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] Error converting {file_name}:")
            print(e.stderr)

    print(f"\nconversion complete. {success_count}/{len(md_files)} files processed.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    # Configuration
    PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
    TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/DMUpapertemplate/template_fixed.docx")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/docx_output")
    FILTER_PATH = os.path.join(PROJECT_ROOT, "tools/thesis_style.lua")
    BIB_PATH = os.path.join(PROJECT_ROOT, "thesis_paper/manuscript_gpt_review/latex/manuscript/references.bib")
    
    # Files to convert
    MD_FILES = [
        "thesis_paper/manuscript_gpt_review/chapter0_abstract.md",
        "thesis_paper/manuscript_gpt_review/chapter0_notation.md",
        "thesis_paper/manuscript_gpt_review/chapter1.md",
        "thesis_paper/manuscript_gpt_review/chapter2.md",
        "thesis_paper/manuscript_gpt_review/chapter3.md",
        "thesis_paper/manuscript_gpt_review/chapter4.md",
        "thesis_paper/manuscript_gpt_review/chapter5.md",
        "thesis_paper/manuscript_gpt_review/chapter6.md",
        "thesis_paper/manuscript_gpt_review/chapter7.md",
        "thesis_paper/manuscript_gpt_review/chapter8.md",
        "thesis_paper/manuscript_gpt_review/chapter9.md",
        "thesis_paper/manuscript_gpt_review/symbol_checklist.md",
        "thesis_paper/manuscript_gpt_review/template.md",
        "thesis_paper/manuscript_gpt_review/writing_checklist.md"
    ]
    
    # Resolve absolute paths
    ABS_MD_FILES = [os.path.join(PROJECT_ROOT, f) for f in MD_FILES]
    
    convert_md_to_docx(ABS_MD_FILES, TEMPLATE_PATH, OUTPUT_DIR, FILTER_PATH, BIB_PATH)
```

---

## 2. modify_template.py (模板样式修复)
路径: `tools/modify_template.py`

```python
import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
import os

PROJECT_ROOT = "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full"
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
```

---

## 3. thesis_style.lua (Pandoc 过滤器)
路径: `tools/thesis_style.lua`

```lua
-- State to track if we are inside a references section
local in_references = false

function Header(el)
  local text = pandoc.utils.stringify(el)
  in_references = false
  
  -- Header supports attributes directly
  if not el.attributes then el.attributes = {} end

  if el.level == 1 then
    if text:match("摘要") or text:match("Abstract") then
      el.attributes['custom-style'] = '摘要题目'
    elseif text:match("参考文献") or text:match("References") then
      el.attributes['custom-style'] = '参考文献标题'
      in_references = true 
    elseif text:match("致谢") or text:match("Acknowledgements") then
      el.attributes['custom-style'] = '致谢'
    else
      el.attributes['custom-style'] = '样式 标题 1 + 段后: 1 行'
    end
  elseif el.level == 2 then
    el.attributes['custom-style'] = 'Heading 2'
    if text:match("参考文献") or text:match("References") then in_references = true end
  elseif el.level == 3 then
    el.attributes['custom-style'] = 'Heading 3'
  end
  return el
end

-- Helper to apply style to blocks inside a list
-- For lists, we often want to keep them as is, but if we need to style content:
local function style_list_content(el, style_name)
  return pandoc.walk_block(el, {
    Para = function(p)
      -- Para doesn't have attributes, so we must replace it with a Div(Plain)
      return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = style_name})
    end,
    Plain = function(p)
      return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = style_name})
    end
  })
end

function BulletList(el)
  if in_references then
     return style_list_content(el, '参考文献正文')
  end
  return el
end

function OrderedList(el)
  if in_references then
     return style_list_content(el, '参考文献正文')
  end
  return el
end

function BlockQuote(el)
  return pandoc.walk_block(el, {
    Para = function(p)
      return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = 'Quote'})
    end
  })
end

function Div(el)
  if el.identifier == 'refs' then
     return pandoc.walk_block(el, {
        Para = function(p) 
           return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = '参考文献正文'})
        end
     })
  end
  return el
end

function Table(el)
  -- Fix caption: Map to '图名中文'
  if el.caption and el.caption.long then
      el.caption.long = pandoc.walk_block(pandoc.Div(el.caption.long), {
         Para = function(p) 
            return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = '图名中文'})
         end,
         Plain = function(p) 
            return pandoc.Div({pandoc.Plain(p.content)}, {['custom-style'] = '图名中文'})
         end
      }).content
  end

  local function process_rows(rows)
      if not rows then return nil end
      for _, row in ipairs(rows) do
          for _, cell in ipairs(row.cells) do
              local new_contents = pandoc.List()
              for _, block in ipairs(cell.contents) do
                 if block.tag == "Para" or block.tag == "Plain" then
                    -- Wrap content in Div(Plain) to apply Table Text style
                    new_contents:insert(pandoc.Div({pandoc.Plain(block.content)}, {['custom-style'] = 'Table Text'}))
                 elseif block.tag == "Div" then
                    if block.attributes and block.attributes['custom-style'] == '正文1' then
                       block.attributes['custom-style'] = 'Table Text'
                    end
                    new_contents:insert(block)
                 else
                    new_contents:insert(block)
                 end
              end
              cell.contents = new_contents
          end
      end
      return rows
  end

  if el.head then process_rows(el.head.rows) end
  if el.bodies then
    for _, body in ipairs(el.bodies) do process_rows(body.rows) end
  end
  if el.foot then process_rows(el.foot.rows) end
  
  return el
end

function Para(el)
  -- Para does not support attributes. 
  -- We must return a Div containing a Plain block with the original content.
  -- This ensures correct styling in Docx without creating nested paragraphs or type errors.

  -- 1. Check for Images -> Figure Caption
  local has_image = false
  pandoc.walk_block(el, { Image = function(_) has_image = true end })
  if has_image then
     return pandoc.Div({pandoc.Plain(el.content)}, {['custom-style'] = '图名中文'})
  end

  -- 2. Check for References
  if in_references then
     return pandoc.Div({pandoc.Plain(el.content)}, {['custom-style'] = '参考文献正文'})
  end

  -- 3. Check for Display Math -> Formula
  local has_display_math = false
  pandoc.walk_block(el, {
    Math = function(m) 
      if m.mathtype == 'DisplayMath' then has_display_math = true end
    end
  })
  if has_display_math then
     return pandoc.Div({pandoc.Plain(el.content)}, {['custom-style'] = 'Formula'})
  end

  -- 4. Standard Body Text
  -- Default style for all other paragraphs
  return pandoc.Div({pandoc.Plain(el.content)}, {['custom-style'] = '正文1'})
end
```
