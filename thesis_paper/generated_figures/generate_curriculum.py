
from layout_engine import LayoutEngine

# Initialize Engine
# A wider layout for sequential timeline
# Canvas Width: 18
# Centers: Stage 1 (3.0), Stage 2 (9.0), Stage 3 (15.0)
layout = LayoutEngine(figsize=(18, 9), grid_size=(18, 9))

# --- Timeline ---
# Draw timeline axis
layout.ax.arrow(1, 1, 16, 0, head_width=0.2, head_length=0.3, fc='#333333', ec='#333333', linewidth=2)
layout.ax.text(17.5, 1, "Epochs", ha='center', va='center', fontsize=12, fontweight='bold')

# Markers aligned with stages
y_marker_top = 1.2
y_marker_line = 1.0
layout.ax.plot([3.0, 3.0], [y_marker_line, y_marker_top], color='#333333', linewidth=2)
layout.ax.text(3.0, 0.6, "Start\n(Spatial)", ha='center', va='top', fontsize=10)

layout.ax.plot([9.0, 9.0], [y_marker_line, y_marker_top], color='#333333', linewidth=2)
layout.ax.text(9.0, 0.6, "Mid-Term\n(Temporal)", ha='center', va='top', fontsize=10)

layout.ax.plot([15.0, 15.0], [y_marker_line, y_marker_top], color='#333333', linewidth=2)
layout.ax.text(15.0, 0.6, "End\n(Joint)", ha='center', va='top', fontsize=10)


# --- Stage 1: Spatial Pre-training (Center X=3.0) ---
S1_X = 3.0
layout.add_node('s1_title', S1_X, 7.5, 3.5, 0.8, "Stage 1", "Spatial Pre-training", style='highlight')
layout.add_node('s1_enc',   S1_X, 5.5, 2.0, 1.2, "Encoder", "Active", style='default')
layout.add_node('s1_temp',  S1_X, 4.0, 2.0, 1.2, "Temporal", "Frozen", style='input')
layout.add_node('s1_dec',   S1_X, 2.5, 2.0, 1.2, "Decoder", "Active", style='default')

# Connections S1 (Vertical)
layout.connect('s1_enc', 's1_temp', style='-', color='#AAAAAA')
layout.connect('s1_temp', 's1_dec', style='-', color='#AAAAAA')


# --- Stage 2: Temporal Evolution (Center X=9.0) ---
S2_X = 9.0
layout.add_node('s2_title', S2_X, 7.5, 3.5, 0.8, "Stage 2", "Temporal Learning", style='highlight')
layout.add_node('s2_enc',   S2_X, 5.5, 2.0, 1.2, "Encoder", "Frozen", style='input')
layout.add_node('s2_temp',  S2_X, 4.0, 2.0, 1.2, "Temporal", "Active", style='default')
layout.add_node('s2_dec',   S2_X, 2.5, 2.0, 1.2, "Decoder", "Frozen", style='input')

# Connections S2 (Vertical)
layout.connect('s2_enc', 's2_temp', style='-', color='#333333')
layout.connect('s2_temp', 's2_dec', style='-', color='#333333')


# --- Stage 3: Joint Optimization (Center X=15.0) ---
S3_X = 15.0
layout.add_node('s3_title', S3_X, 7.5, 3.5, 0.8, "Stage 3", "Joint Optimization", style='highlight')
layout.add_node('s3_enc',   S3_X, 5.5, 2.0, 1.2, "Encoder", "Fine-tune", style='default')
layout.add_node('s3_temp',  S3_X, 4.0, 2.0, 1.2, "Temporal", "Fine-tune", style='default')
layout.add_node('s3_dec',   S3_X, 2.5, 2.0, 1.2, "Decoder", "Fine-tune", style='default')

# Connections S3 (Vertical)
layout.connect('s3_enc', 's3_temp')
layout.connect('s3_temp', 's3_dec')

# Consistency Loop S3
layout.add_node('s3_loss', S3_X + 2.0, 2.5, 1.2, 0.8, "L_dc", "Loss", style='loss')
layout.connect('s3_dec', 's3_loss', start_side='right', end_side='left', style='->')


# --- Transitions (Weights Transfer) ---
# S1 -> S2
layout.connect('s1_enc', 's2_enc', start_side='right', end_side='left', style='->', curve=0, color='#F57F17', label="Weights")
layout.connect('s1_temp', 's2_temp', start_side='right', end_side='left', style='->', curve=0, color='#F57F17')
layout.connect('s1_dec', 's2_dec', start_side='right', end_side='left', style='->', curve=0, color='#F57F17')

# S2 -> S3
layout.connect('s2_enc', 's3_enc', start_side='right', end_side='left', style='->', curve=0, color='#F57F17', label="Weights")
layout.connect('s2_temp', 's3_temp', start_side='right', end_side='left', style='->', curve=0, color='#F57F17')
layout.connect('s2_dec', 's3_dec', start_side='right', end_side='left', style='->', curve=0, color='#F57F17')

# Title
layout.add_title("Figure 3-3: Sequential Curriculum Learning Strategy")

# Save
layout.save('thesis_paper/generated_figures/fig3-3_curriculum.svg')
