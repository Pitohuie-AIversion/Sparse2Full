import graphviz
import os

def generate_unified_operator_figure():
    dot = graphviz.Digraph(comment='Unified Observation Operator Module', format='svg')
    
    # Global Attributes
    # Same as previous files (Times-Roman, 12pt, compact margins, concentrate)
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='0.6', ranksep='0.6', 
             newrank='true', dpi='300', size='6,10', overlap='false', concentrate='true')
    
    # Node Defaults
    dot.attr('node', shape='box', style='rounded,filled', fontname='Times-Roman', fontsize='12', 
             margin='0.15,0.1')
    
    # Edge Defaults
    dot.attr('edge', fontname='Times-Roman', fontsize='11', penwidth='1.5')

    # Colors (Material Design palette matched to previous figures)
    style_blue = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'penwidth': '2.0'} # Data
    style_orange = {'fillcolor': '#fff3e0', 'color': '#e65100', 'penwidth': '2.0', 'shape': 'component'} # Operator
    style_purple = {'fillcolor': '#f3e5f5', 'color': '#6a1b9a', 'penwidth': '2.0'} # Steps (using stage_style from before)
    style_red = {'fillcolor': '#ffebee', 'color': '#c62828', 'style': 'dashed,filled', 'penwidth': '2.0'} # Loss
    style_green = {'fillcolor': '#e8f5e9', 'color': '#2e7d32', 'penwidth': '2.0'} # Mask/Active
    style_grey = {'fillcolor': '#f5f5f5', 'color': '#9e9e9e', 'style': 'dashed,filled', 'fontcolor': '#616161'} # Noise/Frozen
    style_plain = {'shape': 'plain', 'style': 'none', 'fillcolor': 'none', 'color': 'none', 'fontname': 'Times-Roman'}

    # ---------------------------------------------------------
    # Part (a): Unified Observation Operator Principle
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_A') as a:
        a.attr(label='(a) Unified Observation Operator Principle', fontname='Times-Roman', fontsize='14', 
               style='filled', color='#eeeeee', margin='15')
        
        # Left Column: Generation
        with a.subgraph(name='cluster_A_Left') as al:
            al.attr(style='invis')
            al.node('u', 'High-Res Field u', **style_blue)
            al.node('H', 'Observation Operator H\n(Generation-time)', **style_orange)
            al.node('y', 'Sparse Observation y', **style_blue)
            
            al.edge('u', 'H')
            al.edge('H', 'y')

        # Right Column: Training
        with a.subgraph(name='cluster_A_Right') as ar:
            ar.attr(style='invis')
            ar.node('u_hat', 'Prediction û', **style_blue)
            ar.node('DC', 'Training Degradation DC\n(Mirrored Implementation)', **style_orange)
            ar.node('y_hat', 'Reprojected Obs. ŷ', **style_blue)
            
            ar.edge('u_hat', 'DC')
            ar.edge('DC', 'y_hat')

        # Alignment & Connection between H and DC
        with a.subgraph() as s:
            s.attr(rank='same')
            s.node('H'); s.node('DC')
            # Connection
            s.edge('H', 'DC', label='DC ≡ H', style='dashed', dir='none', fontname='Times-Roman', fontsize='12', minlen='2')

        # Noise
        # Position Noise to the right of H/DC row
        a.node('Noise', 'Optional Noise η', **style_grey)
        
        # Loss box
        a.node('Ldc', 'Observation Consistency\nLdc(y, ŷ)', **style_red)
        
        # Align Noise with H/DC? Or higher?
        # User: "Right (top half right) ... dashed line down to observation comparison line"
        # Let's align Noise with H/DC for neatness
        with a.subgraph() as s:
            s.attr(rank='same')
            s.node('Noise')
            # Invisible edge to position Noise right of DC
            # s.edge('DC', 'Noise', style='invis', minlen='1') # This might cause layout issues with clusters
        
        # Align y, y_hat, Ldc
        with a.subgraph() as s:
            s.attr(rank='same')
            s.node('y'); s.node('y_hat'); s.node('Ldc')
        
        # Connections to Loss
        # y -> Ldc (dashed)
        # y_hat -> Ldc (solid)
        a.edge('y', 'Ldc', style='dashed', constraint='false', xlabel='y')
        a.edge('y_hat', 'Ldc', style='solid', constraint='false', xlabel='ŷ')
        
        # Noise connection
        # Connect Noise to the 'y' node or the edge?
        # Graphviz can't connect to edges easily.
        # Let's connect Noise to y (the observation).
        a.edge('Noise', 'y', style='dashed', label='+')

    # ---------------------------------------------------------
    # Part (b): Task-Specific Operator Paths
    # ---------------------------------------------------------
    with dot.subgraph(name='cluster_B') as b:
        b.attr(label='(b) Task-Specific Operator Paths', fontname='Times-Roman', fontsize='14', 
               style='filled', color='#eeeeee', margin='15')
        
        # SR Branch
        with b.subgraph(name='cluster_SR') as sr:
            sr.attr(label='SR branch', style='dashed', color='#aaaaaa', fontcolor='#616161', margin='15')
            
            sr.node('u_sr', 'Input u', **style_blue)
            sr.node('Blur', 'Gaussian Blur\n(Anti-aliasing)', **style_purple)
            sr.node('Down', 'Area Downsample\n(scale = s)', **style_purple)
            sr.node('y_sr', 'Output ySR', **style_blue)
            
            sr.edge('u_sr', 'Blur')
            sr.edge('Blur', 'Down')
            sr.edge('Down', 'y_sr')
            
            # Config Text
            sr_config = 'Config:\n• kernel size k\n• std σ\n• reflection padding\n• area interpolation'
            sr.node('SR_Conf', sr_config, **style_plain, fontcolor='#616161', fontsize='11', align='left')
            
            # Align Config with Blur
            with sr.subgraph() as s:
                s.attr(rank='same')
                s.node('Blur'); s.node('SR_Conf')
                s.edge('Blur', 'SR_Conf', style='invis', minlen='1')

        # Crop Branch
        with b.subgraph(name='cluster_Crop') as cr:
            cr.attr(label='Crop branch', style='dashed', color='#aaaaaa', fontcolor='#616161', margin='15')
            
            cr.node('u_cr', 'Input u', **style_blue)
            cr.node('Crop', 'Center Crop\n(size = hc × wc)', **style_purple)
            cr.node('Sync', 'Binary Mask Sync\n(Mij ∈ {0,1})', **style_purple)
            cr.node('y_cr', 'Output yCrop', **style_blue)
            cr.node('Mask', 'Mask M', **style_green)
            
            cr.edge('u_cr', 'Crop')
            cr.edge('Crop', 'Sync')
            cr.edge('Sync', 'y_cr')
            cr.edge('Sync', 'Mask', constraint='false', style='solid') 
            
            # Align Mask with y_cr
            with cr.subgraph() as s:
                s.attr(rank='same')
                s.node('y_cr'); s.node('Mask')
            
            # Config Text
            cr_config = 'Config:\n• centered geometry\n• patch-aligned size\n• same mask rule in train/test'
            cr.node('CR_Conf', cr_config, **style_plain, fontcolor='#616161', fontsize='11', align='left')
            
            # Align Config with Crop
            with cr.subgraph() as s:
                s.attr(rank='same')
                s.node('Crop'); s.node('CR_Conf')
                s.edge('Crop', 'CR_Conf', style='invis', minlen='1')

    # Force Part A above Part B
    dot.edge('y', 'u_sr', style='invis')
    dot.edge('y_hat', 'u_cr', style='invis')

    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_2_unified_operator'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.svg")

if __name__ == "__main__":
    generate_unified_operator_figure()
