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
