
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'monospace'

# Canvas Setup
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors - Systematic Flow Palette
COLOR_BG = '#F5F5F7'      # Off-white background
COLOR_NODE_FILL = '#FFFFFF' # White nodes
COLOR_NODE_EDGE = '#2C3E50' # Dark Slate Blue edges
COLOR_TEXT = '#34495E'      # Slate text
COLOR_SR = '#E67E22'        # Carrot Orange for SR path
COLOR_CROP = '#16A085'      # Green Sea for Crop path
COLOR_ARROW = '#7F8C8D'     # Asbestos Grey for arrows
COLOR_LABEL_BG = '#ECF0F1'  # Light Grey for label bg

# Function to draw nodes
def draw_node(ax, x, y, w, h, text, subtext=None, color_edge=COLOR_NODE_EDGE, color_fill=COLOR_NODE_FILL):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=color_edge, facecolor=color_fill)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text, 
            ha='center', va='center', fontsize=10, fontweight='bold', color=COLOR_TEXT)
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.15, subtext, 
                ha='center', va='center', fontsize=8, color=COLOR_TEXT, style='italic')
    return (x, y, w, h)

# Function to draw paths
def draw_path(ax, x1, y1, x2, y2, color=COLOR_ARROW, style='-|>', curve=0, text=None):
    connection_style = f"arc3,rad={curve}"
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), 
                                    arrowstyle=style, mutation_scale=15, 
                                    linewidth=1.5, color=color, connectionstyle=connection_style)
    ax.add_patch(arrow)
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + (0.2 if curve > 0 else -0.2)
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=8, 
                bbox=dict(facecolor=COLOR_LABEL_BG, edgecolor='none', pad=2), color=COLOR_TEXT)

# --- Layout ---

# Title
ax.text(6, 7.5, "Unified Observation Operator Mechanism", ha='center', fontsize=14, fontweight='bold', color=COLOR_NODE_EDGE)
ax.text(6, 7.2, "Algorithm 1: H(u) -> y_hat", ha='center', fontsize=10, style='italic', color=COLOR_ARROW)

# Source Node (High Res)
draw_node(ax, 1, 3.5, 1.5, 1.0, "High Res", "u (GT)")

# Branch Split Point
split_x = 3.5
split_y = 4.0

# Path to split
draw_path(ax, 2.6, 4.0, split_x, split_y, style='-', color=COLOR_NODE_EDGE)

# --- Branch A: Super Resolution (Top) ---
# Path Up
draw_path(ax, split_x, split_y, 4.5, 5.5, style='->', color=COLOR_SR, curve=0.2, text="SR Mode")

# Gaussian Blur Node
draw_node(ax, 5, 5.0, 2.0, 1.0, "Gaussian Blur", "Anti-aliasing\nσ, k=5", color_edge=COLOR_SR)

# Path to Downsample
draw_path(ax, 7.1, 5.5, 8.0, 5.5, style='->', color=COLOR_SR)

# Downsample Node
draw_node(ax, 8, 5.0, 2.0, 1.0, "Downsample", "Inter_Area / Cubic\nScale s", color_edge=COLOR_SR)

# Path to Output
draw_path(ax, 10.1, 5.5, 11.0, 4.2, style='->', color=COLOR_SR, curve=-0.2)


# --- Branch B: Crop / Inpainting (Bottom) ---
# Path Down
draw_path(ax, split_x, split_y, 5.0, 2.5, style='->', color=COLOR_CROP, curve=-0.2, text="Crop Mode")

# Center Crop Node
draw_node(ax, 5, 2.0, 2.0, 1.0, "Center Crop", "(h_c, w_c)\nAlign Patch", color_edge=COLOR_CROP)

# Path to Mask
draw_path(ax, 7.1, 2.5, 8.0, 2.5, style='->', color=COLOR_CROP)

# Mask Gen Node
draw_node(ax, 8, 2.0, 2.0, 1.0, "Mask Gen", "Bernoulli / Block\nMask M", color_edge=COLOR_CROP)

# Path to Output
draw_path(ax, 10.1, 2.5, 11.0, 3.8, style='->', color=COLOR_CROP, curve=0.2)


# --- Convergence ---
# Output Node (Low Res Observation)
draw_node(ax, 11, 3.5, 1.5, 1.0, "Observation", "y_hat")


# --- Annotations / Legend ---
# Comparison Cross (Visual vs Nearest) - Conceptual
ax.text(8, 6.5, "vs. Nearest Neighbor", ha='center', fontsize=9, color='#C0392B')
ax.text(8, 6.2, "(Aliasing Artifacts)", ha='center', fontsize=8, color='#C0392B')
draw_path(ax, 8, 6.1, 8, 6.0, style='-[', color='#C0392B') # Bracket pointing down? No just text is fine.

# Consistency Marker
rect_consist = patches.Rectangle((4.8, 1.5), 5.4, 5.0, linewidth=1, edgecolor='#BDC3C7', facecolor='none', linestyle='--')
ax.add_patch(rect_consist)
ax.text(7.5, 1.2, "Configured by YAML (Consistent H & DC)", ha='center', fontsize=9, color='#7F8C8D', style='italic')

plt.tight_layout()
plt.savefig('thesis_paper/generated_figures/fig3-2_observation_operator.svg', dpi=300, bbox_inches='tight')
plt.close()
