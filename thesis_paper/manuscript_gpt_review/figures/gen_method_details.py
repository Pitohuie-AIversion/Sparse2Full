import graphviz
import os

def generate_detail_figures():
    generate_sequential_training_flowchart()
    generate_triple_loss_architecture()

def generate_sequential_training_flowchart():
    """Generates Figure 3-2: Sequential Spatiotemporal Training Strategy Flowchart"""
    dot = graphviz.Digraph(comment='Sequential Training Strategy', format='png')
    dot.attr(rankdir='LR', compound='true', splines='ortho', nodesep='0.5', ranksep='0.5')
    
    # Global node styles
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12')
    
    # Styles
    frozen_style = {'fillcolor': '#eeeeee', 'color': '#bdbdbd', 'fontcolor': '#757575', 'style': 'dashed,filled'}
    active_style = {'fillcolor': '#e8f5e9', 'color': '#2e7d32', 'penwidth': '2.0'}
    data_style = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'shape': 'ellipse'}
    
    # Stage 1: Spatial Pretrain
    with dot.subgraph(name='cluster_S1') as s1:
        s1.attr(label='Stage 1: Spatial Pretrain\n(Focus: Reconstruction)', style='filled', color='#f5f5f5')
        s1.node('In_S1', 'Single Frame y_t', **data_style)
        s1.node('Spatial_S1', 'Spatial Encoder/Decoder', **active_style)
        s1.node('Temp_S1', 'Temporal Module', **frozen_style)
        s1.node('Out_S1', 'Reconstruction u_t', **data_style)
        
        s1.edge('In_S1', 'Spatial_S1')
        s1.edge('Spatial_S1', 'Out_S1', label='Direct Path')
        s1.edge('Spatial_S1', 'Temp_S1', style='invis') # Layout helper

    # Stage 2: Temporal Pretrain
    with dot.subgraph(name='cluster_S2') as s2:
        s2.attr(label='Stage 2: Temporal Pretrain\n(Focus: Dynamics)', style='filled', color='#f5f5f5')
        s2.node('In_S2', 'Seq y_{1:T}', **data_style)
        s2.node('Spatial_S2', 'Spatial Encoder', **frozen_style)
        s2.node('Temp_S2', 'Temporal Module\n(ARWrapper)', **active_style)
        s2.node('Out_S2', 'Latent Evolution z_t', **data_style)
        
        s2.edge('In_S2', 'Spatial_S2')
        s2.edge('Spatial_S2', 'Temp_S2', label='Latent Features')
        s2.edge('Temp_S2', 'Out_S2')

    # Stage 3: Joint Finetuning
    with dot.subgraph(name='cluster_S3') as s3:
        s3.attr(label='Stage 3: Joint Finetuning\n(Focus: Long-term Stability)', style='filled', color='#f5f5f5')
        s3.node('In_S3', 'Long Seq y_{1:T}', **data_style)
        s3.node('Joint_S3', 'Full Model', **active_style)
        s3.node('Out_S3', 'Rollout Pred û_{1:T}', **data_style)
        s3.node('Reg_S3', 'Temporal Regularization\n(Deriv + Energy)', shape='note', fillcolor='#fff9c4', color='#fbc02d')
        
        s3.edge('In_S3', 'Joint_S3')
        s3.edge('Joint_S3', 'Out_S3')
        s3.edge('Out_S3', 'Reg_S3', style='dashed')

    # Edges between stages to show progression
    dot.edge('Out_S1', 'In_S2', style='invis')
    dot.edge('Out_S2', 'In_S3', style='invis')

    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_2_sequential_training'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.png")

def generate_triple_loss_architecture():
    """Generates Figure 3-3: Triple Consistency Loss Architecture"""
    dot = graphviz.Digraph(comment='Triple Consistency Loss', format='png')
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='0.6', ranksep='0.6')
    
    # Global node styles
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12')
    
    # Styles
    tensor_style = {'fillcolor': '#e1f5fe', 'color': '#01579b', 'shape': 'box'}
    op_style = {'fillcolor': '#fff3e0', 'color': '#ff6f00', 'shape': 'component'}
    loss_style = {'fillcolor': '#ffebee', 'color': '#b71c1c', 'shape': 'ellipse', 'penwidth': '2.0'}
    
    # Main Flow
    with dot.subgraph(name='cluster_flow') as c:
        c.attr(label='Forward Pass & Loss Calculation', style='filled', color='#f5f5f5')
        
        c.node('Pred_Z', 'Model Output\n(z-score domain) û⁽ᶻ⁾', **tensor_style)
        c.node('GT_Z', 'Ground Truth\n(z-score domain) u⁽ᶻ⁾', **tensor_style)
        
        c.node('InvNorm', 'Inverse Normalization\nσ·x + μ', **op_style)
        
        c.node('Pred_Phys', 'Prediction\n(Physical Domain) ũ', **tensor_style)
        c.node('GT_Phys', 'Ground Truth\n(Physical Domain) u', **tensor_style) # Implicit
        
        c.node('DC_Op', 'Degradation Operator\nDC ≡ H', **op_style)
        c.node('Pred_Obs', 'Degraded Prediction\nH(ũ)', **tensor_style)
        c.node('GT_Obs', 'Real Observation\ny', **tensor_style)

        # Flow connections
        c.edge('Pred_Z', 'InvNorm')
        c.edge('InvNorm', 'Pred_Phys')
        c.edge('Pred_Phys', 'DC_Op')
        c.edge('DC_Op', 'Pred_Obs')

    # Losses
    dot.node('L_rec', 'L_rec\n||û⁽ᶻ⁾ - u⁽ᶻ⁾||²', **loss_style)
    dot.node('L_spec', 'L_spec\n||FFT(û⁽ᶻ⁾) - FFT(u⁽ᶻ⁾)||²', **loss_style)
    dot.node('L_dc', 'L_dc\n||H(ũ) - y||²', **loss_style)
    
    # Loss Connections
    # L_rec
    dot.edge('Pred_Z', 'L_rec')
    dot.edge('GT_Z', 'L_rec', style='dashed')
    
    # L_spec (FFT implicit in connection)
    dot.edge('Pred_Z', 'L_spec', label='FFT_low')
    dot.edge('GT_Z', 'L_spec', style='dashed')
    
    # L_dc
    dot.edge('Pred_Obs', 'L_dc')
    dot.edge('GT_Obs', 'L_dc', style='dashed')

    output_path = 'thesis_paper/manuscript_gpt_review/figures/fig_3_3_triple_loss'
    dot.render(output_path, cleanup=True)
    print(f"Figure generated at {output_path}.png")

if __name__ == "__main__":
    generate_detail_figures()
