import graphviz
import os

# --- Configuration: Academic Style ---
STYLE = {
    'fontname': 'Helvetica',
    'fontsize': '10',
    'rankdir': 'TB',
    'nodesep': '0.5',
    'ranksep': '0.5',
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

def gen_thesis_structure_logical():
    dot = graphviz.Digraph(comment='Thesis Logical Flow', format='svg')
    apply_style(dot)
    dot.attr(rankdir='TB', compound='true')

    # 1. Introduction (Motivation)
    with dot.subgraph(name='cluster_Intro') as c:
        c.attr(label='Chapter 1: Introduction\n(Research Motivation)', **COLORS['chapter'])
        c.node('Background', 'Background:\nSparse Observation\nin Digital Twins', **COLORS['content'])
        c.node('Gap', 'Gap:\nOperator Mismatch &\nOptimization Dilemma', **COLORS['content'])
        c.edge('Background', 'Gap')

    # 2. Problem & Theory (Foundation)
    with dot.subgraph(name='cluster_Theory') as c:
        c.attr(label='Chapter 2: Problem Formulation\n(Theoretical Foundation)', **COLORS['chapter'])
        c.node('Model', 'Math Model:\nInverse Problem y=H(u)', **COLORS['content'])
        c.node('Analysis', 'Analysis:\nIll-posedness & Consistency', **COLORS['content'])
        c.edge('Model', 'Analysis')

    # 3. Methodology (Solution)
    with dot.subgraph(name='cluster_Method') as c:
        c.attr(label='Chapter 3: Methodology\n(Consistency-First Framework)', **COLORS['chapter'])
        
        # Core Modules
        c.node('Operator', 'Unified Operator\n(H ≡ DC)', **COLORS['content'])
        c.node('Training', 'Sequential Training\n(Spatial->Temporal)', **COLORS['content'])
        c.node('Loss', 'Tri-Component Loss\n(Rec + Spec + DC)', **COLORS['content'])
        
        # Internal logic
        c.edge('Operator', 'Training', style='invis')
        c.edge('Training', 'Loss', style='invis')

    # 4. Experiments (Verification)
    with dot.subgraph(name='cluster_Exp') as c:
        c.attr(label='Chapter 4: Experiments\n(Verification & Analysis)', **COLORS['chapter'])
        c.node('Setup', 'Setup:\nPDEBench (SWE/DRD)', **COLORS['content'])
        c.node('Results', 'Results:\nAccuracy / Robustness / Efficiency', **COLORS['content'])
        c.edge('Setup', 'Results')

    # 5. Conclusion
    with dot.subgraph(name='cluster_Concl') as c:
        c.attr(label='Chapter 5: Conclusion', **COLORS['chapter'])
        c.node('Summary', 'Summary & Outlook', **COLORS['content'])

    # --- Logical Flow Edges ---
    
    # Gap -> Theory
    dot.edge('Gap', 'Model', label='Formulate')
    
    # Theory -> Method
    dot.edge('Analysis', 'Operator', label='Guide Design')
    
    # Method -> Exp
    dot.edge('Loss', 'Setup', label='Evaluate')
    
    # Exp -> Conclusion
    dot.edge('Results', 'Summary', label='Synthesize')

    # Feedback Loop (Optional, to show iteration)
    # dot.edge('Results', 'Method', style='dashed', label='Refine', constraint='false')

    path = os.path.join(OUTPUT_DIR, 'fig1-1_thesis_structure')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

if __name__ == "__main__":
    gen_thesis_structure_logical()
