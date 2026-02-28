import graphviz
import os

def generate_teaser_figure():
    # Create a directed graph
    dot = graphviz.Digraph(comment='Evaluation Consistency First Framework', format='svg')
    # Optimized layout: TB flow with horizontal internal sub-flows using rank=same
    # Increased spacing (nodesep/ranksep) to prevent label/line overlap
    # size="6,10": Constrain width to 6 inches, allow height up to 10 inches
    # dpi="300": High resolution
    # concentrate="true": Merge parallel edges to reduce clutter
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='1.0', ranksep='1.2', newrank='true', size='6,10', overlap='false', dpi='300', concentrate='true')
    
    # Global node styles
    # Added margin to nodes to give text breathing room inside boxes
    # Fixed height/width ratio to make boxes look more consistent
    # Xiaosi (Small 4) is 12pt
    dot.attr('node', shape='box', style='rounded,filled', fontname='Times-Roman', fontsize='12', margin='0.25,0.15')
    
    # Styles for different types of nodes
    # Lighter background colors for better contrast
    data_style = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'penwidth': '2.0'} # Blue 50
    process_style = {'fillcolor': '#fff3e0', 'color': '#e65100', 'penwidth': '2.0', 'shape': 'component'} # Orange 50
    loss_style = {'fillcolor': '#ffebee', 'color': '#c62828', 'style': 'dashed,filled', 'penwidth': '2.0'} # Red 50
    stage_style = {'fillcolor': '#f3e5f5', 'color': '#6a1b9a', 'penwidth': '2.0'} # Purple 50

    # Module A: Observation
    with dot.subgraph(name='cluster_A') as c:
        c.attr(label='(a) Unified Observation Generation', style='filled', color='#eeeeee', margin='10')
        c.node('GT', 'Input: Ground Truth u(x,t)', **data_style)
        c.node('H_Op', 'Observation Operator H\n(Anti-aliasing + Interp)', **process_style)
        c.node('Obs', 'Output: Sparse Observation y(x,t)', **data_style)
        c.node('Noise', 'Noise n', shape='plain')
        
        # Use headlabel/taillabel instead of xlabel for better placement control
        # or increase minlen to force edge length
        c.edge('GT', 'H_Op', xlabel='u', minlen='2')
        c.edge('H_Op', 'Obs', xlabel='H(u)', minlen='2')
        c.edge('Noise', 'Obs', xlabel='+ n', minlen='2')
        
        # Force horizontal layout for this module
        with c.subgraph() as s:
            s.attr(rank='same')
            s.node('GT'); s.node('H_Op'); s.node('Obs'); s.node('Noise')

    # Module B: Sequential Training
    with dot.subgraph(name='cluster_B') as c:
        c.attr(label='(b) Sequential Spatiotemporal Training', style='filled', color='#eeeeee', margin='20')
        
        # Stage 1
        with c.subgraph(name='cluster_S1') as s1:
            s1.attr(label='Stage 1: Spatial Pretrain', style='dashed', color='#aaaaaa', margin='15')
            s1.node('SpatialNet', 'Spatial Encoder/Decoder', **stage_style)
            s1.node('Rec_S', 'Output: Spatial Rec. u_s', **data_style)
            s1.edge('SpatialNet', 'Rec_S', xlabel='u_s = F_s(y)', minlen='3')
            
            # Force horizontal
            with s1.subgraph() as s:
                s.attr(rank='same')
                s.node('SpatialNet'); s.node('Rec_S')

        # Stage 2
        with c.subgraph(name='cluster_S2') as s2:
            s2.attr(label='Stage 2: Temporal Pretrain', style='dashed', color='#aaaaaa', margin='15')
            s2.node('TempNet', 'Temporal Evolution', **stage_style)
            s2.node('Feat_T', 'Output: Evolved Features', **data_style)
            s2.edge('TempNet', 'Feat_T', xlabel='z_{t+1} = F_t(z_t)', minlen='3')
            
            # Force horizontal
            with s2.subgraph() as s:
                s.attr(rank='same')
                s.node('TempNet'); s.node('Feat_T')

        # Stage 3
        with c.subgraph(name='cluster_S3') as s3:
            s3.attr(label='Stage 3: Joint Fine-tuning', style='dashed', color='#aaaaaa', margin='15')
            s3.node('JointNet', 'Joint Model', **stage_style)
            s3.node('Pred_Final', 'Final Output: Prediction û', **data_style)
            s3.edge('JointNet', 'Pred_Final', xlabel='û = F(y)', minlen='3')
            
            # Force horizontal
            with s3.subgraph() as s:
                s.attr(rank='same')
                s.node('JointNet'); s.node('Pred_Final')
        
        # Enforce sequential order S1 -> S2 -> S3 (Vertical stack of horizontal rows)
        c.edge('Rec_S', 'TempNet', style='invis')
        c.edge('Feat_T', 'JointNet', style='invis')

    # Module C: Losses
    with dot.subgraph(name='cluster_C') as c:
        c.attr(label='(c) Triple Consistency Loss', style='filled', color='#eeeeee', margin='10')
        
        c.node('DC_Op', 'Training Degradation DC ≡ H\n(Mirrored Implementation)', **process_style)
        c.node('Pred_Obs', 'H(û)', **data_style)
        
        c.node('L_rec', 'L_rec: Reconstruction', **loss_style)
        c.node('L_spec', 'L_spec: Spectral Consistency', **loss_style)
        c.node('L_dc', 'L_dc: Observation Consistency', **loss_style)
        
        c.edge('Pred_Final', 'DC_Op')
        c.edge('DC_Op', 'Pred_Obs')

        # Horizontal alignment
        with c.subgraph() as s:
            s.attr(rank='same')
            s.node('DC_Op'); s.node('Pred_Obs')

        with c.subgraph() as s:
            s.attr(rank='same')
            s.node('L_rec'); s.node('L_spec'); s.node('L_dc')
            
        c.edge('Pred_Obs', 'L_dc')

    # Module D: Evaluation
    with dot.subgraph(name='cluster_D') as c:
        c.attr(label='(d) Physical Evaluation', style='filled', color='#eeeeee', margin='10')
        c.node('Eval', 'Metrics:\nRel-L2, H_err, Energy Spectrum', shape='note', fillcolor='#fff9c4', color='#fbc02d')

    # Connections between clusters
    # Use constraint=False or style adjustments to reduce wire clutter
    dot.edge('Obs', 'SpatialNet', lhead='cluster_S1', constraint='true')
    dot.edge('Obs', 'JointNet', lhead='cluster_S3', constraint='false', style='dashed') # Reduced visual weight for skip connection
    
    # Loss connections - optimize routing
    dot.edge('Pred_Final', 'L_rec', constraint='true')
    dot.edge('GT', 'L_rec', constraint='false', style='dotted') # Dotted for reference data
    
    dot.edge('Pred_Final', 'L_spec', constraint='true')
    dot.edge('GT', 'L_spec', constraint='false', style='dotted')
    
    dot.edge('Pred_Obs', 'L_dc', constraint='true')
    dot.edge('Obs', 'L_dc', constraint='false', style='dotted') # Long skip connection made dotted
    
    # Evaluation connection
    dot.edge('Pred_Final', 'Eval')

    # Render
    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_1_framework'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.png")

if __name__ == "__main__":
    generate_teaser_figure()
