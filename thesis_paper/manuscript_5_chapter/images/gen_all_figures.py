import graphviz
import os

# --- Configuration: Academic Style ---
STYLE = {
    'fontname': 'Helvetica',
    'fontsize': '12',   # Unified font size
    'rankdir': 'LR',
    'nodesep': '0.5',
    'ranksep': '0.6',
    'splines': 'ortho',  # Orthogonal edges for clean diagrams
}

# Color Palette (Professional/Nature-inspired)
COLORS = {
    'data': {'fillcolor': '#E3F2FD', 'color': '#1565C0', 'fontcolor': '#0D47A1'},       # Light Blue / Dark Blue
    'process': {'fillcolor': '#F3E5F5', 'color': '#7B1FA2', 'fontcolor': '#4A148C'},    # Light Purple / Dark Purple
    'operator': {'fillcolor': '#E8F5E9', 'color': '#2E7D32', 'fontcolor': '#1B5E20'},   # Light Green / Dark Green
    'loss': {'fillcolor': '#FFEBEE', 'color': '#C62828', 'fontcolor': '#B71C1C'},       # Light Red / Dark Red
    'frozen': {'fillcolor': '#EEEEEE', 'color': '#9E9E9E', 'fontcolor': '#616161', 'style': 'dashed,filled'}, # Grey
    'group': {'style': 'filled', 'color': '#F5F5F5', 'fontcolor': '#424242'},           # Light Grey Background
}

OUTPUT_DIR = "thesis_paper/manuscript_5_chapter/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_style(dot):
    # Use dpi=300 for high quality rasterization (even for SVG, good for preview)
    dot.attr(dpi='300')
    
    dot.attr(fontname=STYLE['fontname'], fontsize=STYLE['fontsize'])
    dot.attr('node', shape='box', style='rounded,filled', 
             fontname=STYLE['fontname'], fontsize=STYLE['fontsize'], penwidth='1.5')
    dot.attr('edge', fontname=STYLE['fontname'], fontsize=STYLE['fontsize'], penwidth='1.2')

def get_node_attr(type_key):
    base = {'shape': 'box', 'style': 'rounded,filled'}
    base.update(COLORS.get(type_key, {}))
    return base

# --- Figure 3-1: Framework Overview ---
def gen_fig3_1():
    dot = graphviz.Digraph(comment='Consistency-First Framework', format='svg')
    apply_style(dot)
    dot.attr(rankdir='LR', compound='true') # Left-to-Right flow
    
    # 1. Input Construction
    with dot.subgraph(name='cluster_Input') as c:
        c.attr(label='(a) Input Construction', **COLORS['group'])
        c.node('y', 'Sparse Obs\n(y)', **get_node_attr('data'))
        c.node('mask', 'Mask\n(M)', **get_node_attr('data'))
        c.node('coords', 'Coords\n(x,t)', **get_node_attr('data'))
        c.node('Xin', 'Input Tensor\n(Concat)', shape='point', width='0.1')
        
        c.edge('y', 'Xin')
        c.edge('mask', 'Xin')
        c.edge('coords', 'Xin')

    # 2. Backbone
    with dot.subgraph(name='cluster_Model') as c:
        c.attr(label='(b) Spatiotemporal Model', **COLORS['group'])
        c.node('Encoder', 'Spatial Encoder\n(CNN/Trans)', **get_node_attr('process'))
        c.node('Temporal', 'Temporal Evolution\n(Latent)', **get_node_attr('process'))
        c.node('Decoder', 'Decoder\n(Reconstruction)', **get_node_attr('process'))
        
        c.edge('Xin', 'Encoder')
        c.edge('Encoder', 'Temporal')
        c.edge('Temporal', 'Decoder')
        c.node('u_hat', 'Reconstruction\n(û)', **get_node_attr('data'))
        c.edge('Decoder', 'u_hat')

    # 3. Consistency Loop
    with dot.subgraph(name='cluster_Loop') as c:
        c.attr(label='(c) Consistency Loop', **COLORS['group'])
        
        # Upper Path: Reconstruction Loss
        c.node('u_gt', 'Ground Truth\n(u)', **get_node_attr('data'))
        c.node('L_rec', 'L_rec', **get_node_attr('loss'))
        
        # Lower Path: Observation Consistency
        c.node('DC', 'Unified Operator\n(DC ≡ H)', **get_node_attr('operator'))
        c.node('y_hat', 'Reprojection\n(ŷ)', **get_node_attr('data'))
        c.node('L_dc', 'L_dc', **get_node_attr('loss'))
        
        # Spectral Path
        c.node('FFT', 'FFT', shape='plain')
        c.node('L_spec', 'L_spec', **get_node_attr('loss'))

    # Connections
    # Reconstruction
    dot.edge('u_hat', 'L_rec')
    dot.edge('u_gt', 'L_rec')
    
    # Consistency
    dot.edge('u_hat', 'DC')
    dot.edge('DC', 'y_hat')
    dot.edge('y_hat', 'L_dc')
    dot.edge('y', 'L_dc', style='dashed', label='Constraint') # From Input
    
    # Spectral
    dot.edge('u_hat', 'FFT', style='dashed')
    dot.edge('u_gt', 'FFT', style='dashed')
    dot.edge('FFT', 'L_spec')

    path = os.path.join(OUTPUT_DIR, 'fig3-1_framework')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

# --- Figure 3-2: Unified Operator ---
def gen_fig3_2():
    dot = graphviz.Digraph(comment='Unified Operator', format='svg')
    apply_style(dot)
    dot.attr(rankdir='TB')
    
    dot.node('Input', 'High-Res Field\n(u)', **get_node_attr('data'))
    
    # Branch SR
    with dot.subgraph(name='cluster_SR') as c:
        c.attr(label='Task: Super-Resolution (SR)', **COLORS['group'])
        c.node('Kernel', 'Gaussian Kernel\n(Anti-aliasing)', **get_node_attr('operator'))
        c.node('Down', 'Downsample\n(Area Interp)', **get_node_attr('operator'))
        c.edge('Input', 'Kernel')
        c.edge('Kernel', 'Down')
        c.node('Out_SR', 'Low-Res Obs\n(y_sr)', **get_node_attr('data'))
        c.edge('Down', 'Out_SR')

    # Branch Crop
    with dot.subgraph(name='cluster_Crop') as c:
        c.attr(label='Task: Sparse Crop', **COLORS['group'])
        c.node('Center', 'Center Align\n(Patch Grid)', **get_node_attr('operator'))
        c.node('MaskGen', 'Mask Generation\n(Binary M)', **get_node_attr('operator'))
        c.edge('Input', 'Center')
        c.edge('Center', 'MaskGen')
        c.node('Out_Crop', 'Masked Obs\n(y_crop, M)', **get_node_attr('data'))
        c.edge('MaskGen', 'Out_Crop')

    path = os.path.join(OUTPUT_DIR, 'fig3-2_operator')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

# --- Figure 3-3: Sequential Training ---
def gen_fig3_3():
    dot = graphviz.Digraph(comment='Sequential Training', format='svg')
    apply_style(dot)
    dot.attr(rankdir='LR')

    # Stage 1
    with dot.subgraph(name='cluster_S1') as c:
        c.attr(label='Stage 1: Spatial', **COLORS['group'])
        c.node('S1_Enc', 'Spatial Enc', **get_node_attr('process'))
        c.node('S1_Temp', 'Temporal\n(Frozen)', **get_node_attr('frozen'))
        c.node('S1_Dec', 'Decoder', **get_node_attr('process'))
        c.edge('S1_Enc', 'S1_Dec')

    # Stage 2
    with dot.subgraph(name='cluster_S2') as c:
        c.attr(label='Stage 2: Temporal', **COLORS['group'])
        c.node('S2_Enc', 'Spatial Enc\n(Frozen)', **get_node_attr('frozen'))
        c.node('S2_Temp', 'Temporal\n(Active)', **get_node_attr('process'))
        c.node('S2_Dec', 'Decoder\n(Frozen)', **get_node_attr('frozen'))
        c.edge('S2_Enc', 'S2_Temp')
        c.edge('S2_Temp', 'S2_Dec')

    # Stage 3
    with dot.subgraph(name='cluster_S3') as c:
        c.attr(label='Stage 3: Joint', **COLORS['group'])
        c.node('S3_All', 'Full Model\n(All Active)', **get_node_attr('process'))
        c.node('L_joint', 'Physics Loss\n(L_total)', **get_node_attr('loss'))
        c.edge('S3_All', 'L_joint')

    # Flow
    dot.edge('S1_Dec', 'S2_Enc', style='dashed', label='Weights')
    dot.edge('S2_Dec', 'S3_All', style='dashed', label='Weights')

    path = os.path.join(OUTPUT_DIR, 'fig3-3_sequential')
    dot.render(path, cleanup=True)
    print(f"Generated: {path}.svg")

if __name__ == "__main__":
    gen_fig3_1()
    gen_fig3_2()
    gen_fig3_3()
