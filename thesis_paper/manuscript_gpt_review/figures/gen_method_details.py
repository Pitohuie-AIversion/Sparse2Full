import graphviz
import os

def generate_detail_figures():
    generate_sequential_training_flowchart()
    generate_triple_loss_architecture()

def generate_sequential_training_flowchart():
    """Generates Figure 3-2: Sequential Spatiotemporal Training Strategy Flowchart"""
    dot = graphviz.Digraph(comment='Sequential Training Strategy', format='svg')
    # Optimized layout: TB for portrait page
    # Increased spacing to prevent overlap
    # DPI=300, concentrate edges
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='1.0', ranksep='1.2', newrank='true', size='6,10', overlap='false', dpi='300', concentrate='true')
    
    # Global node styles
    # Added margin to nodes
    # Xiaosi (Small 4) is 12pt
    dot.attr('node', shape='box', style='rounded,filled', fontname='Times-Roman', fontsize='12', margin='0.25,0.15')
    
    # Styles
    # Use standard material colors (50-100 range for bg, 700-900 for text/border)
    # Styles mapped to fig_3_1 for consistency
    data_style = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'penwidth': '2.0'} # Blue 50
    process_style = {'fillcolor': '#fff3e0', 'color': '#e65100', 'penwidth': '2.0', 'shape': 'component'} # Orange 50
    loss_style = {'fillcolor': '#ffebee', 'color': '#c62828', 'style': 'dashed,filled', 'penwidth': '2.0'} # Red 50
    stage_style = {'fillcolor': '#f3e5f5', 'color': '#6a1b9a', 'penwidth': '2.0'} # Purple 50
    
    # Legacy mappings
    frozen_style = {'fillcolor': '#f5f5f5', 'color': '#9e9e9e', 'fontcolor': '#616161', 'style': 'dashed,filled'}
    active_style = stage_style # Map active modules to stage style (Purple)
    
    # Stage 1: Spatial Pretrain
    with dot.subgraph(name='cluster_S1') as s1:
        s1.attr(label='Stage 1: Spatial Pretrain\n(Focus: Reconstruction)', style='filled', color='#eeeeee', margin='15')
        s1.node('In_S1', 'Single Frame y_t', **data_style)
        s1.node('Spatial_S1', 'Spatial Encoder/Decoder', **active_style)
        s1.node('Temp_S1', 'Temporal Module', **frozen_style)
        s1.node('Out_S1', 'Reconstruction u_t', **data_style)
        
        # Vertical flow inside stage
        s1.edge('In_S1', 'Spatial_S1', minlen='1')
        s1.edge('Spatial_S1', 'Out_S1', xlabel='Direct Path', minlen='1')
        s1.edge('Spatial_S1', 'Temp_S1', style='invis') 

    # Stage 2: Temporal Pretrain
    with dot.subgraph(name='cluster_S2') as s2:
        s2.attr(label='Stage 2: Temporal Pretrain\n(Focus: Dynamics)', style='filled', color='#eeeeee', margin='15')
        s2.node('In_S2', 'Seq y_{1:T}', **data_style)
        s2.node('Spatial_S2', 'Spatial Encoder', **frozen_style)
        s2.node('Temp_S2', 'Temporal Module\n(ARWrapper)', **active_style)
        s2.node('Out_S2', 'Latent Evolution z_t', **data_style)
        
        # Vertical flow inside stage
        s2.edge('In_S2', 'Spatial_S2', minlen='1')
        s2.edge('Spatial_S2', 'Temp_S2', xlabel='Latent Features', minlen='1')
        s2.edge('Temp_S2', 'Out_S2', minlen='1')

    # Stage 3: Joint Finetuning
    with dot.subgraph(name='cluster_S3') as s3:
        s3.attr(label='Stage 3: Joint Finetuning\n(Focus: Long-term Stability)', style='filled', color='#eeeeee', margin='15')
        s3.node('In_S3', 'Long Seq y_{1:T}', **data_style)
        s3.node('Joint_S3', 'Full Model', **active_style)
        s3.node('Out_S3', 'Rollout Pred û_{1:T}', **data_style)
        s3.node('Reg_S3', 'Temporal Regularization\n(Deriv + Energy)', **process_style)
        
        # Vertical flow inside stage
        s3.edge('In_S3', 'Joint_S3', minlen='1')
        s3.edge('Joint_S3', 'Out_S3', minlen='1')
        s3.edge('Out_S3', 'Reg_S3', style='dashed', minlen='1')

    # Force vertical column layout: S1 | S2 | S3
    # Use invisible edges to constrain relative positions
    # rank=same for top nodes creates the columns
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('In_S1'); s.node('In_S2'); s.node('In_S3')

    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_2_sequential_training'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.png")

def generate_triple_loss_architecture():
    """Generates Figure 3-3: Triple Consistency Loss Architecture"""
    dot = graphviz.Digraph(comment='Triple Consistency Loss', format='svg')
    # Optimized layout: TB for portrait page
    # Compact layout: Reduced spacing
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='0.6', ranksep='0.6', newrank='true', size='6,8', overlap='false', dpi='300', concentrate='true')
    
    # Global node styles
    # Smaller margins for compactness
    # Xiaosi (Small 4) is 12pt
    dot.attr('node', shape='box', style='rounded,filled', fontname='Times-Roman', fontsize='12', margin='0.15,0.1')
    
    # Styles
    data_style = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'penwidth': '2.0'} # Blue 50
    process_style = {'fillcolor': '#fff3e0', 'color': '#e65100', 'penwidth': '2.0', 'shape': 'component'} # Orange 50
    loss_style = {'fillcolor': '#ffebee', 'color': '#c62828', 'style': 'dashed,filled', 'penwidth': '2.0'} # Red 50
    
    # Styles mapped to fig_3_1 for consistency
    tensor_style = data_style
    op_style = process_style
    
    # Main Flow
    with dot.subgraph(name='cluster_flow') as c:
        c.attr(label='Forward Pass & Loss Calculation', style='filled', color='#eeeeee', margin='15')
        
        c.node('Pred_Z', 'Model Output\n(z-score domain) û⁽ᶻ⁾', **tensor_style)
        c.node('GT_Z', 'Ground Truth\n(z-score domain) u⁽ᶻ⁾', **tensor_style)
        
        c.node('InvNorm', 'Inverse Normalization\nσ·x + μ', **op_style)
        
        c.node('Pred_Phys', 'Prediction\n(Physical Domain) ũ', **tensor_style)
        c.node('GT_Phys', 'Ground Truth\n(Physical Domain) u', **tensor_style) # Implicit
        
        c.node('DC_Op', 'Degradation Operator\nDC ≡ H', **op_style)
        c.node('Pred_Obs', 'Degraded Prediction\nH(ũ)', **tensor_style)
        c.node('GT_Obs', 'Real Observation\ny', **tensor_style)

        # Flow connections
        # Reduced minlen for compactness
        c.edge('Pred_Z', 'InvNorm', minlen='1')
        c.edge('InvNorm', 'Pred_Phys', minlen='1')
        c.edge('Pred_Phys', 'DC_Op', minlen='1')
        c.edge('DC_Op', 'Pred_Obs', minlen='1')

    # Losses
    dot.node('L_rec', 'L_rec\n||û⁽ᶻ⁾ - u⁽ᶻ⁾||²', **loss_style)
    dot.node('L_spec', 'L_spec\n||FFT(û⁽ᶻ⁾) - FFT(u⁽ᶻ⁾)||²', **loss_style)
    dot.node('L_dc', 'L_dc\n||H(ũ) - y||²', **loss_style)
    
    # Loss Connections
    # L_rec
    dot.edge('Pred_Z', 'L_rec', minlen='1')
    dot.edge('GT_Z', 'L_rec', style='dashed', minlen='1')
    
    # L_spec (FFT implicit in connection)
    dot.edge('Pred_Z', 'L_spec', xlabel='FFT_low', minlen='1')
    dot.edge('GT_Z', 'L_spec', style='dashed', minlen='1')
    
    # L_dc
    dot.edge('Pred_Obs', 'L_dc', minlen='1')
    dot.edge('GT_Obs', 'L_dc', style='dashed', minlen='1')

    # --- Layout Optimization: Force Alignment ---
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Pred_Z'); s.node('GT_Z')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Pred_Phys'); s.node('GT_Phys')

    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Pred_Obs'); s.node('GT_Obs')
    # ------------------------------------------

    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_3_triple_loss'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.png")

if __name__ == "__main__":
    generate_detail_figures()
