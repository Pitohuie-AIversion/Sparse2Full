import graphviz
import os

# --- Configuration: Academic Style (Chinese) ---
# Assuming 'Noto Serif CJK SC' as a substitute for SimSun if SimSun is not available.
# Graphviz relies on fontconfig. 'SimSun' is standard Songti.
FONT_NAME = 'SimSun' 
# Fallback to Noto Serif CJK SC if SimSun is strictly not found, but usually SimSun is mapped or installed.
# Using 'Noto Serif CJK SC' based on fc-list result to be safe.
FONT_NAME = 'Noto Serif CJK SC' 

STYLE = {
    'fontname': FONT_NAME,
    'fontsize': '10',
    'rankdir': 'TB',
    'nodesep': '0.3',   # Tighter spacing
    'ranksep': '0.4',
    'splines': 'ortho',
}

# Color Palette (Professional/Nature-inspired - Lighter for print)
COLORS = {
    'chapter': {'fillcolor': '#E3F2FD', 'color': '#1565C0', 'fontcolor': '#000000', 'shape': 'box', 'style': 'rounded,filled'},
    'content': {'fillcolor': '#FFFFFF', 'color': '#616161', 'fontcolor': '#000000', 'shape': 'box', 'style': 'dashed'},
}

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_style(dot):
    dot.attr(fontname=STYLE['fontname'], fontsize=STYLE['fontsize'])
    dot.attr('node', shape='box', style='rounded,filled', 
             fontname=STYLE['fontname'], fontsize=STYLE['fontsize'], penwidth='1.0')
    dot.attr('edge', fontname=STYLE['fontname'], fontsize='9', penwidth='1.0')

def gen_thesis_structure_cn():
    # Set size to fit A4 text width (approx 6 inches / 15cm)
    # dpi=300 for high quality
    dot = graphviz.Digraph(comment='Thesis Logical Flow CN', format='svg')
    dot.attr(dpi='300')
    dot.attr(size='6,8') # Width, Height limit
    
    apply_style(dot)
    dot.attr(rankdir='TB', compound='true')

    # 1. Introduction
    with dot.subgraph(name='cluster_Intro') as c:
        c.attr(label='第一章：绪论\n(研究动机)', **COLORS['chapter'])
        c.node('Background', '研究背景：\n数字孪生中的稀疏观测难题', **COLORS['content'])
        c.node('Gap', '研究现状与挑战：\n算子失配与优化困境', **COLORS['content'])
        c.edge('Background', 'Gap')

    # 2. Problem & Theory
    with dot.subgraph(name='cluster_Theory') as c:
        c.attr(label='第二章：问题建模与理论分析\n(理论基础)', **COLORS['chapter'])
        c.node('Model', '数学建模：\n稀疏观测反问题 y=H(u)', **COLORS['content'])
        c.node('Analysis', '理论分析：\n适定性分析与一致性误差界', **COLORS['content'])
        c.edge('Model', 'Analysis')

    # 3. Methodology
    with dot.subgraph(name='cluster_Method') as c:
        c.attr(label='第三章：算法设计与实现\n(核心方法 Consistency-First)', **COLORS['chapter'])
        
        # Use a hidden structure to layout nodes horizontally or vertically as needed
        # For a flow chart, vertical stack is usually better for A4 portrait
        c.node('Operator', '统一观测算子构建\n(H ≡ DC)', **COLORS['content'])
        c.node('Training', '序列化时空课程学习\n(空间->时序->联合)', **COLORS['content'])
        c.node('Loss', '三元混合损失函数\n(重建+谱+观测一致性)', **COLORS['content'])
        
        c.edge('Operator', 'Training')
        c.edge('Training', 'Loss')

    # 4. Experiments
    with dot.subgraph(name='cluster_Exp') as c:
        c.attr(label='第四章：实验结果与分析\n(验证与评估)', **COLORS['chapter'])
        c.node('Setup', '实验设置：\nPDEBench数据集 (SWE/DRD)', **COLORS['content'])
        c.node('Results', '多维结果分析：\n重建精度 / 频谱保真度 / 计算效率', **COLORS['content'])
        c.edge('Setup', 'Results')

    # 5. Conclusion
    with dot.subgraph(name='cluster_Concl') as c:
        c.attr(label='第五章：总结与展望', **COLORS['chapter'])
        c.node('Summary', '全文总结与未来展望', **COLORS['content'])

    # --- Logical Flow Edges ---
    
    # Gap -> Theory
    dot.edge('Gap', 'Model', label='  问题形式化  ')
    
    # Theory -> Method
    dot.edge('Analysis', 'Operator', label='  理论指导设计  ')
    
    # Method -> Exp
    dot.edge('Loss', 'Setup', label='  实验验证  ')
    
    # Exp -> Conclusion
    dot.edge('Results', 'Summary', label='  总结归纳  ')

    path = os.path.join(OUTPUT_DIR, 'fig1-1_thesis_structure')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

if __name__ == "__main__":
    gen_thesis_structure_cn()
