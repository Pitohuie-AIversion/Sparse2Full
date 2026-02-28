
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Canvas Setup
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis('off')

# Colors
COLOR_BOX_FILL = '#E6F3FF'  # Light Blue
COLOR_BOX_EDGE = '#0055AA'  # Strong Blue
COLOR_INPUT_FILL = '#F0F0F0' # Light Grey
COLOR_INPUT_EDGE = '#666666' # Dark Grey
COLOR_GT_FILL = '#E6FFE6'    # Light Green
COLOR_GT_EDGE = '#008800'    # Strong Green
COLOR_ARROW = '#333333'      # Dark Grey for arrows
COLOR_LOSS = '#D32F2F'       # Red for loss

def draw_box(ax, x, y, w, h, text, color_fill, color_edge, subtext=None):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=color_edge, facecolor=color_fill)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.2 if subtext else 0), text, 
            ha='center', va='center', fontsize=12, fontweight='bold', color=color_edge)
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.2, subtext, 
                ha='center', va='center', fontsize=10, color=color_edge)
    return (x, y, w, h)

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, style='-|>', curve=0):
    connection_style = f"arc3,rad={curve}"
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), 
                                    arrowstyle=style, mutation_scale=20, 
                                    linewidth=2, color=color, connectionstyle=connection_style)
    ax.add_patch(arrow)

# --- 1. Inputs (Left) ---
draw_box(ax, 1, 5, 2, 1.5, "Input", COLOR_INPUT_FILL, COLOR_INPUT_EDGE, "Sparse y\nMask M\nCoords")

# --- 2. Model Pipeline (Center) ---
# Encoder
draw_box(ax, 4, 5, 2, 1.5, "Encoder", COLOR_BOX_FILL, COLOR_BOX_EDGE, "Spatial Feat")
# Temporal
draw_box(ax, 7, 5, 2, 1.5, "Temporal", COLOR_BOX_FILL, COLOR_BOX_EDGE, "Evolution")
# Decoder
draw_box(ax, 10, 5, 2, 1.5, "Decoder", COLOR_BOX_FILL, COLOR_BOX_EDGE, "Reconstruction u")

# Arrows for pipeline
draw_arrow(ax, 3.2, 5.75, 3.9, 5.75) # Input -> Encoder
draw_arrow(ax, 6.2, 5.75, 6.9, 5.75) # Encoder -> Temporal
draw_arrow(ax, 9.2, 5.75, 9.9, 5.75) # Temporal -> Decoder

# --- 3. Consistency Loop (Right & Bottom) ---
# DC Operator (Observation Operator H)
draw_box(ax, 10, 2, 2, 1.0, "DC Operator", COLOR_INPUT_FILL, COLOR_INPUT_EDGE, "H(u) -> y_hat")

# Arrow from Decoder down to DC
draw_arrow(ax, 11, 5, 11, 3.2) # Decoder -> DC

# Comparison with Input (Consistency Loss)
draw_box(ax, 4, 2, 2, 1.0, "Consistency\nCheck", '#FFF0F0', COLOR_LOSS, "L_dc = ||y - y_hat||")

# Arrow from DC to Check
draw_arrow(ax, 10, 2.5, 6.0, 2.5) # DC -> Check

# Arrow from Input to Check (down and right)
draw_arrow(ax, 2, 5, 2, 2.5) # Input -> down
draw_arrow(ax, 2, 2.5, 4.0, 2.5) # ... -> Check

# --- 4. Ground Truth & Other Losses (Top) ---
# Move GT up to avoid crowding
draw_box(ax, 10, 7.5, 2, 0.8, "Ground Truth", COLOR_GT_FILL, COLOR_GT_EDGE, "Full Field u_gt")

# Reconstruction Loss Node (Pixel-wise) - Placed to the right
# Shifted left slightly to stay within canvas (14.0)
draw_box(ax, 12.0, 6.0, 1.5, 0.8, "L_rec", '#FFF0F0', COLOR_LOSS, "Pixel-wise")
# Arrow from GT (Right side)
draw_arrow(ax, 12.0, 7.9, 12.75, 6.8, style='-', color=COLOR_LOSS, curve=-0.2) 
# Arrow from Decoder (Right side)
draw_arrow(ax, 12.0, 5.75, 12.75, 6.0, style='-', color=COLOR_LOSS, curve=0.2)

# Spectral Loss Node (Frequency) - Placed in the gap above Temporal
draw_box(ax, 7.5, 6.8, 1.5, 0.8, "L_spec", '#FFF0F0', COLOR_LOSS, "FFT Low-Freq")
# Arrow from GT (Left side)
draw_arrow(ax, 10.0, 7.9, 9.0, 7.2, curve=0.2, style='->', color=COLOR_LOSS) 
# Arrow from Decoder (Top side)
draw_arrow(ax, 10.5, 6.5, 8.25, 6.8, curve=0.3, style='->', color=COLOR_LOSS)

# --- 5. Labels & Annotations ---
# Main Title
ax.text(7, 0.5, "Figure 3-1: Consistency-First Reconstruction Framework", 
        ha='center', fontsize=16, fontweight='bold', color='#333333')

# Highlight "Golden Rule"
rect_golden = patches.Rectangle((3.8, 1.5), 8.5, 2.0, linewidth=2, edgecolor='#FFD700', facecolor='none', linestyle='--')
ax.add_patch(rect_golden)
ax.text(8, 1.2, "Golden Rule: Training DC == Observation H", ha='center', fontsize=12, color='#B8860B', fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('thesis_paper/generated_figures/fig3-1_framework.svg', dpi=300, bbox_inches='tight')
plt.close()
