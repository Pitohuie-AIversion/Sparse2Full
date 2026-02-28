import graphviz
import os

# --- Configuration: Academic Style ---
STYLE = {
    'fontname': 'Helvetica',
    'fontsize': '10',
    'rankdir': 'TB',
    'nodesep': '0.5',
    'ranksep': '0.6',
    'splines': 'ortho',
}

# Color Palette (Professional/Nature-inspired)
COLORS = {
    'chapter': {'fillcolor': '#E3F2FD', 'color': '#1565C0', 'fontcolor': '#0D47A1', 'shape': 'box', 'style': 'rounded,filled'},
    'content': {'fillcolor': '#FFFFFF', 'color': '#616161', 'fontcolor': '#424242', 'shape': 'box', 'style': 'dashed'},
}

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_style(dot):
    dot.attr(fontname=STYLE['fontname'], fontsize=STYLE['fontsize'])
    dot.attr('node', shape='box', style='rounded,filled', 
             fontname=STYLE['fontname'], fontsize=STYLE['fontsize'], penwidth='1.5')
    dot.attr('edge', fontname=STYLE['fontname'], fontsize='9', penwidth='1.2')

def gen_thesis_structure():
    dot = graphviz.Digraph(comment='Thesis Structure', format='svg')
    apply_style(dot)
    dot.attr(rankdir='TB', compound='true')

    # Chapter 1
    with dot.subgraph(name='cluster_C1') as c1:
        c1.attr(label='', style='invis')
        c1.node('C1', '第1章 绪论\n(Introduction)', **COLORS['chapter'])
        c1.node('C1_content', '研究背景与意义\n国内外现状综述\n研究内容与创新点', **COLORS['content'])
        c1.edge('C1', 'C1_content', style='invis')

    # Chapter 2
    with dot.subgraph(name='cluster_C2') as c2:
        c2.attr(label='', style='invis')
        c2.node('C2', '第2章 问题建模与理论分析\n(Problem Formulation)', **COLORS['chapter'])
        c2.node('C2_content', '稀疏观测数学模型\n统一观测算子定义\n一致性误差界推导', **COLORS['content'])
        c2.edge('C2', 'C2_content', style='invis')

    # Chapter 3
    with dot.subgraph(name='cluster_C3') as c3:
        c3.attr(label='', style='invis')
        c3.node('C3', '第3章 算法设计与实现\n(Methodology)', **COLORS['chapter'])
        c3.node('C3_content', 'Consistency-First 框架\n网络架构设计\n序列化训练流程\n三元混合损失函数', **COLORS['content'])
        c3.edge('C3', 'C3_content', style='invis')

    # Chapter 4
    with dot.subgraph(name='cluster_C4') as c4:
        c4.attr(label='', style='invis')
        c4.node('C4', '第4章 实验结果与分析\n(Experiments)', **COLORS['chapter'])
        c4.node('C4_content', 'PDEBench 数据集实验\n对比实验与消融实验\n精度/频谱/效率分析', **COLORS['content'])
        c4.edge('C4', 'C4_content', style='invis')

    # Chapter 5
    with dot.subgraph(name='cluster_C5') as c5:
        c5.attr(label='', style='invis')
        c5.node('C5', '第5章 总结与展望\n(Conclusion)', **COLORS['chapter'])
        c5.node('C5_content', '研究工作总结\n方法局限性分析\n未来研究展望', **COLORS['content'])
        c5.edge('C5', 'C5_content', style='invis')

    # Main Flow Edges
    dot.edge('C1', 'C2')
    dot.edge('C2', 'C3')
    dot.edge('C3', 'C4')
    dot.edge('C4', 'C5')

    # Align contents with chapters (visual trick)
    # Actually, simpler to just have edges C1 -> C2 -> ...
    # And have the content nodes hang off them or be part of the cluster.
    # Let's try putting them in the same cluster and forcing order?
    # Graphviz layout is tricky. Let's just link them.
    dot.edge('C1', 'C1_content', dir='none', style='dotted')
    dot.edge('C2', 'C2_content', dir='none', style='dotted')
    dot.edge('C3', 'C3_content', dir='none', style='dotted')
    dot.edge('C4', 'C4_content', dir='none', style='dotted')
    dot.edge('C5', 'C5_content', dir='none', style='dotted')

    path = os.path.join(OUTPUT_DIR, 'fig1-1_thesis_structure')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

if __name__ == "__main__":
    gen_thesis_structure()
