import graphviz
import os

def generate_detail_figures():
    generate_sequential_training_flowchart()
    generate_triple_loss_architecture()

def generate_sequential_training_flowchart():
    """Generates Figure 3-2: Sequential Spatiotemporal Training Strategy Flowchart (Chinese Academic Style)"""
    dot = graphviz.Digraph(comment='Sequential Training Strategy', format='svg')
    # Optimized layout: LR (Left-to-Right) for the 3 stages horizontally as requested
    dot.attr(rankdir='LR', compound='true', splines='ortho', nodesep='0.6', ranksep='0.8', newrank='true', size='10,6', overlap='false', dpi='300', concentrate='true')
    
    # Global node styles: Academic, matching other figures but using Chinese font
    dot.attr('node', shape='box', style='rounded,filled', fontname='SimSun', fontsize='12', margin='0.2,0.1')
    
    # Styles (Matching fig 3-1 and 3-3 Material Design palette)
    data_style = {'fillcolor': '#e3f2fd', 'color': '#1565c0', 'penwidth': '2.0', 'shape': 'box'} # Blue 50
    process_style = {'fillcolor': '#fff3e0', 'color': '#e65100', 'penwidth': '2.0', 'shape': 'component'} # Orange 50
    active_style = {'fillcolor': '#f3e5f5', 'color': '#6a1b9a', 'penwidth': '2.0', 'shape': 'box'} # Purple 50, solid border
    frozen_style = {'fillcolor': '#f5f5f5', 'color': '#9e9e9e', 'fontcolor': '#616161', 'style': 'dashed,filled', 'penwidth': '1.5', 'shape': 'box'} # Grey dashed
    loss_style = {'fillcolor': '#ffebee', 'color': '#c62828', 'penwidth': '2.0', 'style': 'dashed,filled', 'shape': 'box'} # Red 50
    
    # Stage 1: Spatial Pretrain
    with dot.subgraph(name='cluster_S1') as s1:
        s1.attr(label='阶段1：空间预训练\n(Spatial Pretraining)\nBatch: [40, 1, 128, 128]', style='filled', color='#eeeeee', margin='15', fontname='SimSun', fontsize='14')
        s1.node('In_S1', '单帧观测 y_t\n[40, 1, 128, 128]', **data_style)
        s1.node('Spatial_S1', '空间模型 (EDSR)\n[40, 64, 128, 128]', **active_style)
        s1.node('Out_S1', '重建结果 û_t\n[40, 1, 128, 128]', **data_style)
        
        # Explicitly show the frozen temporal module in Stage 1 to emphasize the architectural consistency
        s1.node('Temp_S1', '时序模块\n(Frozen)', **frozen_style)
        
        s1.edge('In_S1', 'Spatial_S1')
        s1.edge('Spatial_S1', 'Out_S1')
        
        # Put Temp_S1 parallel to Spatial_S1 to show it exists but is bypassed/frozen
        s1.edge('Spatial_S1', 'Temp_S1', style='invis')
        with s1.subgraph() as s:
            s.attr(rank='same')
            s.node('Spatial_S1'); s.node('Temp_S1')

    # Stage 2: Temporal Pretrain
    with dot.subgraph(name='cluster_S2') as s2:
        s2.attr(label='阶段2：时序预训练\n(Temporal Pretraining)\nBatch: [4, 10, 1, 128, 128]', style='filled', color='#eeeeee', margin='15', fontname='SimSun', fontsize='14')
        s2.node('In_S2', '序列观测 y_{1:T}\n[4, 10, 1, 128, 128]', **data_style)
        s2.node('Spatial_S2', '空间模型 (EDSR)\n[4, 10, 1, 128, 128]', **frozen_style)
        s2.node('Temp_S2', '时序模型 (VideoSwin)\n[4, 96, 10, 128, 128]', **active_style)
        s2.node('Out_S2', '未来帧预测 û_{T+1}\n[4, 1, 1, 128, 128]', **data_style)
        
        s2.edge('In_S2', 'Spatial_S2')
        s2.edge('Spatial_S2', 'Temp_S2', xlabel=' [4, 10, 1, 128, 128] ')
        s2.edge('Temp_S2', 'Out_S2')

    # Stage 3: Joint Finetuning
    with dot.subgraph(name='cluster_S3') as s3:
        s3.attr(label='阶段3：时空联合微调\n(Joint Fine-tuning)\nBatch: [4, 10, 1, 128, 128]', style='filled', color='#eeeeee', margin='15', fontname='SimSun', fontsize='14')
        s3.node('In_S3', '序列观测 y_{1:T}\n[4, 10, 1, 128, 128]', **data_style)
        s3.node('Joint_S3', '完整时空模型\n(Teacher Forcing 注入)\nEDSR + VideoSwin', **active_style)
        s3.node('Out_S3', '联合预测 û_{T+1}\n[4, 1, 1, 128, 128]', **data_style)

        
        # Loss constraint matching Section 3.5 formula
        loss_label = '联合损失约束\nL_rec + λ_spec L_spec + λ_dc L_dc'
        s3.node('Reg_S3', loss_label, **loss_style)
        
        # Vertical flow inside stage 3
        s3.edge('In_S3', 'Joint_S3')
        s3.edge('Joint_S3', 'Out_S3')
        s3.edge('Out_S3', 'Reg_S3', style='dashed')

    # Horizontal alignment across stages (align the inputs at the top)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('In_S1'); s.node('In_S2'); s.node('In_S3')

    # Progression arrows between stages
    dot.edge('Out_S1', 'In_S2', style='invis')
    dot.edge('Out_S2', 'In_S3', style='invis')
    
    # Legend
    # Place legend at the bottom, horizontally.
    with dot.subgraph(name='cluster_legend') as leg:
        leg.attr(label='图例 (Legend)', style='solid', color='#cccccc', fontname='SimSun', fontsize='12', margin='5', rankdir='LR')
        leg.node('leg_frozen', '灰色虚框：冻结参数', **frozen_style, fontsize='10')
        leg.node('leg_active', '深色实框：当前训练模块', **active_style, fontsize='10')
        leg.node('leg_data', '蓝色框：输入/输出', **data_style, fontsize='10')
        leg.node('leg_loss', '橙色框：损失约束', **loss_style, fontsize='10')
        
        # Horizontal layout for legend
        with leg.subgraph() as s:
            s.attr(rank='same')
            s.node('leg_frozen'); s.node('leg_active'); s.node('leg_data'); s.node('leg_loss')
            s.edge('leg_frozen', 'leg_active', style='invis')
            s.edge('leg_active', 'leg_data', style='invis')
            s.edge('leg_data', 'leg_loss', style='invis')

    # Force legend to bottom
    dot.edge('Reg_S3', 'leg_loss', style='invis')

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
