
from layout_engine import LayoutEngine

# Initialize Engine
layout = LayoutEngine(figsize=(15, 9), grid_size=(15, 9))

# --- 1. Main Pipeline Nodes ---
layout.add_node('input', 2, 5, 2, 1.5, "Input", "Sparse y\nMask M", style='input')
layout.add_node('encoder', 5, 5, 2, 1.5, "Encoder", "Spatial Feat")
layout.add_node('temporal', 8, 5, 2, 1.5, "Temporal", "Evolution")
layout.add_node('decoder', 11, 5, 2, 1.5, "Decoder", "Reconstruction u")

# Connect Pipeline
layout.connect('input', 'encoder')
layout.connect('encoder', 'temporal')
layout.connect('temporal', 'decoder')

# --- 2. Ground Truth & Losses ---
layout.add_node('gt', 11, 7.5, 2, 0.8, "Ground Truth", "u_gt", style='gt')

# L_rec (Pixel-wise) - Auto aligned
layout.add_node('l_rec', 13.5, 6.25, 1.5, 0.8, "L_rec", "Pixel-wise", style='loss')
layout.connect('gt', 'l_rec', start_side='right', end_side='top', curve=-0.2, color='#D32F2F')
layout.connect('decoder', 'l_rec', start_side='right', end_side='bottom', curve=0.2, color='#D32F2F')

# L_spec (Frequency)
layout.add_node('l_spec', 8, 7, 1.5, 0.8, "L_spec", "Low-Freq", style='loss')
layout.connect('gt', 'l_spec', start_side='left', end_side='top', curve=0.2, color='#D32F2F')
layout.connect('decoder', 'l_spec', start_side='top', end_side='bottom', curve=0.3, color='#D32F2F')

# --- 3. Consistency Loop ---
layout.add_node('dc_op', 11, 2.5, 2, 1.0, "DC Operator", "H(u)", style='input')
layout.connect('decoder', 'dc_op', start_side='bottom', end_side='top')

layout.add_node('check', 5, 2.5, 2, 1.0, "Consistency", "L_dc", style='loss')
layout.connect('dc_op', 'check') # Auto right->left
layout.connect('input', 'check', start_side='bottom', end_side='left', curve=0.2)

# Title
layout.add_title("Figure 3-1: Consistency-First Reconstruction Framework (Auto-Layout)")

# Save
layout.save('thesis_paper/generated_figures/fig3-1_framework_auto.svg')
