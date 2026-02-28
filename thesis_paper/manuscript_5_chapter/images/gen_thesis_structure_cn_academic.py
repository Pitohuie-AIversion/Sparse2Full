import graphviz
import os

# --- Configuration: Academic Style (Chinese) ---
# Font: Songti equivalent
FONT_NAME = 'Noto Serif CJK SC' 

STYLE = {
    'fontname': FONT_NAME,
    'fontsize': '12',       # Unified font size
    'rankdir': 'TB',
    'nodesep': '0.5',
    'ranksep': '0.6',
    'splines': 'ortho',     # Orthogonal lines for professional look
}

# Color Palette (Academic/Print-Friendly)
# Black and White / Grayscale with very subtle blue for distinction
COLORS = {
    'chapter_cluster': {'style': 'dashed', 'color': '#424242', 'fontcolor': '#000000', 'penwidth': '0.8'},
    'node_chapter': {
        'shape': 'box', 
        'style': 'filled', 
        'fillcolor': '#E8EAF6', # Very light indigo/grey
        'color': '#1A237E',     # Dark Blue border
        'fontcolor': '#000000',
        'width': '5.0',         # Fixed minimum width (approx 12.7cm) to fit page width
        'height': '0.6'
    },
    'node_content': {
        'shape': 'box', 
        'style': 'filled', 
        'fillcolor': '#FFFFFF', # White
        'color': '#424242',     # Dark Grey border
        'fontcolor': '#424242',
        'width': '4.5',         # Slightly narrower than chapter header
        'fontsize': '12'        # Unified size
    },
    'edge': {
        'color': '#000000',
        'penwidth': '1.0',
        'arrowsize': '0.7'
    }
}

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_style(dot):
    # Use dpi=300 for high quality rasterization
    dot.attr(dpi='300')
    
    dot.attr(fontname=STYLE['fontname'], fontsize=STYLE['fontsize'])
    dot.attr('node', fontname=STYLE['fontname'])
    dot.attr('edge', fontname=STYLE['fontname'], **COLORS['edge'])

def gen_thesis_structure_cn_academic():
    # A4 width is ~8.27 inches. Text width usually ~6 inches.
    # We set node width to ~5 inches to fill the space nicely.
    dot = graphviz.Digraph(comment='Thesis Logical Flow CN Academic', format='svg')
    # DPI 300 for high quality print
    dot.attr(dpi='300')
    
    apply_style(dot)
    dot.attr(rankdir='TB', compound='true')
    
    # 1. Introduction
    with dot.subgraph(name='cluster_Intro') as c:
        c.attr(label='', **COLORS['chapter_cluster'])
        c.node('Ch1', '第一章：绪论 (研究动机)', **COLORS['node_chapter'])
        c.node('Ch1_Detail', '研究背景：数字孪生稀疏观测难题\n现状挑战：算子失配与优化困境', **COLORS['node_content'])
        c.edge('Ch1', 'Ch1_Detail', style='invis') # Force alignment

    # 2. Problem & Theory
    with dot.subgraph(name='cluster_Theory') as c:
        c.attr(label='', **COLORS['chapter_cluster'])
        c.node('Ch2', '第二章：问题建模与理论分析 (理论基础)', **COLORS['node_chapter'])
        c.node('Ch2_Detail', '数学建模：稀疏观测反问题 y=H(u)\n理论分析：适定性分析与一致性误差界', **COLORS['node_content'])
        c.edge('Ch2', 'Ch2_Detail', style='invis')

    # 3. Methodology
    with dot.subgraph(name='cluster_Method') as c:
        c.attr(label='', **COLORS['chapter_cluster'])
        c.node('Ch3', '第三章：算法设计与实现 (核心方法)', **COLORS['node_chapter'])
        c.node('Ch3_Detail', '1. 统一观测算子构建 (H ≡ DC)\n2. 序列化时空课程学习 (空间->时序->联合)\n3. 三元混合损失函数 (重建+谱+观测一致性)', **COLORS['node_content'])
        c.edge('Ch3', 'Ch3_Detail', style='invis')

    # 4. Experiments
    with dot.subgraph(name='cluster_Exp') as c:
        c.attr(label='', **COLORS['chapter_cluster'])
        c.node('Ch4', '第四章：实验结果与分析 (验证与评估)', **COLORS['node_chapter'])
        c.node('Ch4_Detail', '实验设置：PDEBench数据集 (SWE/DRD)\n多维评估：重建精度 / 频谱保真度 / 计算效率', **COLORS['node_content'])
        c.edge('Ch4', 'Ch4_Detail', style='invis')

    # 5. Conclusion
    with dot.subgraph(name='cluster_Concl') as c:
        c.attr(label='', **COLORS['chapter_cluster'])
        c.node('Ch5', '第五章：总结与展望', **COLORS['node_chapter'])
        c.node('Ch5_Detail', '全文总结 / 局限性分析 / 未来展望', **COLORS['node_content'])
        c.edge('Ch5', 'Ch5_Detail', style='invis')

    # --- Logical Flow Edges ---
    # Connect Content to next Chapter
    dot.edge('Ch1_Detail', 'Ch2', label=' 问题形式化 ')
    dot.edge('Ch2_Detail', 'Ch3', label=' 理论指导设计 ')
    dot.edge('Ch3_Detail', 'Ch4', label=' 实验验证 ')
    dot.edge('Ch4_Detail', 'Ch5', label=' 总结归纳 ')

    path = os.path.join(OUTPUT_DIR, 'fig1-1_thesis_structure')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

if __name__ == "__main__":
    gen_thesis_structure_cn_academic()
