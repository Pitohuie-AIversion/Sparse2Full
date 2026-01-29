import graphviz
import os

def generate_teaser_figure():
    # Create a directed graph
    dot = graphviz.Digraph(comment='Evaluation Consistency First Framework', format='png')
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='0.8', ranksep='0.8')
    
    # Global node styles
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12')
    
    # Styles for different types of nodes
    data_style = {'fillcolor': '#e1f5fe', 'color': '#01579b', 'penwidth': '2.0'}
    process_style = {'fillcolor': '#fff3e0', 'color': '#ff6f00', 'penwidth': '2.0', 'shape': 'component'}
    loss_style = {'fillcolor': '#ffebee', 'color': '#b71c1c', 'style': 'dashed,filled', 'penwidth': '2.0'}
    stage_style = {'fillcolor': '#f3e5f5', 'color': '#4a148c', 'penwidth': '2.0'}

    # Module A: Observation
    with dot.subgraph(name='cluster_A') as c:
        c.attr(label='(a) Unified Observation Generation', style='filled', color='#eeeeee')
        c.node('GT', 'Ground Truth u', **data_style)
        c.node('H_Op', 'Observation Operator H\n(Anti-aliasing + Interp)', **process_style)
        c.node('Obs', 'Sparse Observation y', **data_style)
        c.node('Noise', 'Noise n', shape='plain')
        
        c.edge('GT', 'H_Op')
        c.edge('H_Op', 'Obs')
        c.edge('Noise', 'Obs')

    # Module B: Sequential Training
    with dot.subgraph(name='cluster_B') as c:
        c.attr(label='(b) Sequential Spatiotemporal Training', style='filled', color='#eeeeee')
        
        # Stage 1
        with c.subgraph(name='cluster_S1') as s1:
            s1.attr(label='Stage 1: Spatial Pretrain', style='dashed', color='#aaaaaa')
            s1.node('SpatialNet', 'Spatial Encoder/Decoder', **stage_style)
            s1.node('Rec_S', 'Spatial Rec. u_s', **data_style)
            s1.edge('SpatialNet', 'Rec_S')

        # Stage 2
        with c.subgraph(name='cluster_S2') as s2:
            s2.attr(label='Stage 2: Temporal Pretrain', style='dashed', color='#aaaaaa')
            s2.node('TempNet', 'Temporal Evolution', **stage_style)
            s2.node('Feat_T', 'Evolved Features', **data_style)
            s2.edge('TempNet', 'Feat_T')

        # Stage 3
        with c.subgraph(name='cluster_S3') as s3:
            s3.attr(label='Stage 3: Joint Fine-tuning', style='dashed', color='#aaaaaa')
            s3.node('JointNet', 'Joint Model', **stage_style)
            s3.node('Pred_Final', 'Final Prediction û', **data_style)
            s3.edge('JointNet', 'Pred_Final')

    # Module C: Losses
    with dot.subgraph(name='cluster_C') as c:
        c.attr(label='(c) Triple Consistency Loss', style='filled', color='#eeeeee')
        
        c.node('DC_Op', 'Training Degradation DC ≡ H\n(Mirrored Implementation)', **process_style)
        c.node('Pred_Obs', 'H(û)', **data_style)
        
        c.node('L_rec', 'L_rec: Reconstruction', **loss_style)
        c.node('L_spec', 'L_spec: Spectral Consistency', **loss_style)
        c.node('L_dc', 'L_dc: Observation Consistency', **loss_style)
        
        c.edge('Pred_Final', 'DC_Op')
        c.edge('DC_Op', 'Pred_Obs')

    # Module D: Evaluation
    with dot.subgraph(name='cluster_D') as c:
        c.attr(label='(d) Physical Evaluation', style='filled', color='#eeeeee')
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
