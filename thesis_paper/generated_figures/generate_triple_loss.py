
from layout_engine import LayoutEngine

# Initialize Engine
# Canvas Size: 16x10 for balanced vertical/horizontal spread
layout = LayoutEngine(figsize=(16, 10), grid_size=(16, 10))

# --- 1. Sources (Left) ---
# Model Output (u_hat)
layout.add_node('u_hat', 3.0, 7.0, 2.0, 1.2, "Reconstruction", "u_hat", style='default')

# Ground Truth (u)
layout.add_node('u_gt', 3.0, 3.0, 2.0, 1.2, "Ground Truth", "u_gt", style='gt')

# --- 2. Transformation Layers (Middle Column) ---
# FFT Transform (Top Branch)
layout.add_node('fft_hat', 6.0, 8.5, 1.5, 0.8, "FFT", "F(u_hat)", style='input')
layout.add_node('fft_gt',  6.0, 7.5, 1.5, 0.8, "FFT", "F(u_gt)", style='input')

# Reconstruction Loss (Middle Branch)
# Align directly with u_hat/u_gt midpoint (y=5.0)

# DC Operator (Bottom Branch)
layout.add_node('dc_op', 6.0, 2.5, 1.5, 0.8, "DC Op", "H(u_hat)", style='input')
# Observation y (Input)
layout.add_node('obs',   3.0, 1.0, 2.0, 1.2, "Observation", "y (Input)", style='input') # Original input y


# --- 3. Loss Modules (Right Column) ---
# L_spec (Spectral Loss) - Top
layout.add_node('l_spec', 10.0, 8.0, 2.5, 1.2, "L_spec", "||F(u)-F(u_hat)||", style='loss')

# L_rec (Reconstruction Loss) - Middle
layout.add_node('l_rec', 10.0, 5.0, 2.5, 1.2, "L_rec", "||u - u_hat||", style='loss')

# L_dc (Consistency Loss) - Bottom
layout.add_node('l_dc', 10.0, 2.0, 2.5, 1.2, "L_dc", "||y - H(u_hat)||", style='loss')


# --- 4. Total Loss (Far Right) ---
layout.add_node('l_total', 14.0, 5.0, 2.0, 6.0, "Total Loss", "Weighted Sum", style='highlight')


# --- Connections ---

# Top Branch: Spectral
layout.connect('u_hat', 'fft_hat', start_side='top', end_side='left', curve=0.3, style='->')
layout.connect('u_gt',  'fft_gt',  start_side='top', end_side='left', curve=0.5, style='->', color='#008800') # Green for GT path

layout.connect('fft_hat', 'l_spec', start_side='right', end_side='left', style='->')
layout.connect('fft_gt',  'l_spec', start_side='right', end_side='left', style='->', color='#008800')

# Middle Branch: Reconstruction
layout.connect('u_hat', 'l_rec', start_side='right', end_side='left', style='->')
layout.connect('u_gt',  'l_rec', start_side='right', end_side='left', style='->', color='#008800')

# Bottom Branch: Consistency
layout.connect('u_hat', 'dc_op', start_side='bottom', end_side='left', curve=-0.3, style='->')
layout.connect('dc_op', 'l_dc',  start_side='right', end_side='left', style='->')
# Input y connects directly to L_dc (conceptual input source)
layout.connect('obs',   'l_dc',  start_side='right', end_side='left', style='->', color='#666666', curve=0.3)


# Aggregation to Total Loss
layout.connect('l_spec', 'l_total', start_side='right', end_side='left', style='->', label="λ_s")
layout.connect('l_rec',  'l_total', start_side='right', end_side='left', style='->', label="1.0")
layout.connect('l_dc',   'l_total', start_side='right', end_side='left', style='->', label="λ_dc")


# Title
layout.add_title("Figure 3-4: Tri-Component Loss Function Architecture")

# Save
layout.save('thesis_paper/generated_figures/fig3-4_triple_loss.svg')
