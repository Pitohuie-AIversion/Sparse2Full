
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

@dataclass
class BoxStyle:
    fill_color: str
    edge_color: str
    font_color: str = None
    linewidth: float = 2.0
    fontsize: float = 12
    fontweight: str = 'bold'
    boxstyle: str = "round,pad=0.1"

class LayoutEngine:
    def __init__(self, figsize=(16, 10), grid_size=(16, 10), padding=0.5, base_fontsize=12):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.grid_w, self.grid_h = grid_size
        self.ax.set_xlim(0, self.grid_w)
        self.ax.set_ylim(0, self.grid_h)
        self.ax.axis('off')
        self.nodes = {}
        self.padding = padding
        self.base_fontsize = base_fontsize # Unified font size
        
        # Default Styles - All use base_fontsize
        self.styles = {
            'default': BoxStyle('#E6F3FF', '#0055AA', fontsize=base_fontsize),
            'input': BoxStyle('#F0F0F0', '#666666', fontsize=base_fontsize),
            'gt': BoxStyle('#E6FFE6', '#008800', fontsize=base_fontsize),
            'loss': BoxStyle('#FFF0F0', '#D32F2F', fontsize=base_fontsize), # Same size!
            'highlight': BoxStyle('#FFF8E1', '#F57F17', fontsize=base_fontsize)
        }

    def add_node(self, name: str, x: float, y: float, w: float = None, h: float = None, 
                 label: str = "", subtext: str = None, style: str = 'default', anchor: str = 'center',
                 auto_size: bool = True):
        """
        Add a node. If auto_size is True, w and h are ignored (or used as min size) 
        and calculated based on text length.
        """
        s = self.styles.get(style, self.styles['default'])
        font_color = s.font_color if s.font_color else s.edge_color
        
        # --- Auto-Size Calculation ---
        if auto_size:
            # Estimate text width: char_count * approx_char_width
            # A rough heuristic for matplotlib default font at fontsize 12
            # Width factor ~ 0.15 per char per fontsize unit? 
            # Let's say 12pt font -> 0.15 width unit per char
            char_w_factor = 0.08 * (s.fontsize / 10) 
            text_w = len(label) * char_w_factor * 2.0 # Bold is wider
            if subtext:
                sub_w = len(subtext) * char_w_factor * 1.8
                text_w = max(text_w, sub_w)
            
            # Add padding
            calc_w = text_w + 0.6
            calc_h = 1.2 if subtext else 0.8
            
            # Use provided w/h as minimums if given
            w = max(w, calc_w) if w else calc_w
            h = max(h, calc_h) if h else calc_h

        # Adjust x, y based on anchor
        if anchor == 'center':
            x = x - w/2
            y = y - h/2
        
        # Draw
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle=s.boxstyle, 
                                      linewidth=s.linewidth, edgecolor=s.edge_color, facecolor=s.fill_color)
        self.ax.add_patch(rect)
        
        # Text Logic
        cx, cy = x + w/2, y + h/2
        if subtext:
            self.ax.text(cx, cy + 0.15, label, ha='center', va='center', 
                         fontsize=s.fontsize, fontweight=s.fontweight, color=font_color)
            self.ax.text(cx, cy - 0.2, subtext, ha='center', va='center', 
                         fontsize=s.fontsize - 2, color=font_color) # Subtext slightly smaller
        else:
            self.ax.text(cx, cy, label, ha='center', va='center', 
                         fontsize=s.fontsize, fontweight=s.fontweight, color=font_color)

        # Store node info for connection
        self.nodes[name] = {
            'x': x, 'y': y, 'w': w, 'h': h, 
            'cx': cx, 'cy': cy, 
            'top': (cx, y + h),
            'bottom': (cx, y),
            'left': (x, cy),
            'right': (x + w, cy),
            'color': s.edge_color
        }
        return self.nodes[name]

    def _get_connection_point(self, node_name: str, side: str, offset: float = 0.0) -> Tuple[float, float]:
        """
        Get exact connection point with optional offset along the edge.
        side: 'top', 'bottom', 'left', 'right'
        offset: -1.0 to 1.0 (0 is center)
        """
        n = self.nodes[node_name]
        cx, cy = n['cx'], n['cy']
        w, h = n['w'], n['h']
        
        if side == 'top':
            return (cx + offset * (w/2), n['y'] + h)
        elif side == 'bottom':
            return (cx + offset * (w/2), n['y'])
        elif side == 'left':
            return (n['x'], cy + offset * (h/2))
        elif side == 'right':
            return (n['x'] + w, cy + offset * (h/2))
        else:
            return (cx, cy)

    def connect(self, source_name: str, target_name: str, 
                start_side: str = 'auto', end_side: str = 'auto', 
                start_offset: float = 0.0, end_offset: float = 0.0,
                style: str = '-|>', curve: float = 0, color: str = '#333333', label: str = None):
        """
        Auto-connect two nodes. 
        Sides: 'top', 'bottom', 'left', 'right', 'auto'
        """
        n1 = self.nodes[source_name]
        n2 = self.nodes[target_name]
        
        # Auto-detect best connection points if 'auto'
        if start_side == 'auto':
            # Simple heuristic based on relative positions
            dx = n2['cx'] - n1['cx']
            dy = n2['cy'] - n1['cy']
            if abs(dx) > abs(dy): # Horizontal dominance
                start_side = 'right' if dx > 0 else 'left'
            else: # Vertical dominance
                start_side = 'top' if dy > 0 else 'bottom'
                
        if end_side == 'auto':
            # Usually opposite of start, but checking relative pos is better
            dx = n1['cx'] - n2['cx']
            dy = n1['cy'] - n2['cy']
            if abs(dx) > abs(dy):
                end_side = 'right' if dx > 0 else 'left'
            else:
                end_side = 'top' if dy > 0 else 'bottom'

        p1 = self._get_connection_point(source_name, start_side, start_offset)
        p2 = self._get_connection_point(target_name, end_side, end_offset)
        
        # Draw arrow
        connection_style = f"arc3,rad={curve}"
        arrow = patches.FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15, 
                                        linewidth=1.5, color=color, connectionstyle=connection_style)
        self.ax.add_patch(arrow)
        
        # Label on line
        if label:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2 + (0.2 if curve > 0 else -0.2)
            self.ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9, 
                         bbox=dict(facecolor='#FFFFFF', edgecolor='none', alpha=0.8), color=color)

    def add_title(self, text):
        self.ax.text(self.grid_w/2, self.grid_h - 0.5, text, ha='center', fontsize=16, fontweight='bold', color='#333333')

    def save(self, path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

