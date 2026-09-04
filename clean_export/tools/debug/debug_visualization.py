#!/usr/bin/env python3
"""
Debug visualization script to test matplotlib functionality
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def debug_visualization():
    """Debug visualization generation"""
    base_dir = Path(".")
    runs_dir = base_dir / "runs" / "temporal_nar_100epochs"
    output_dir = base_dir / "comprehensive_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Load training data
    training_file = runs_dir / "TemporalNAR-DR2D-128-100epochs-s2025" / "training_history.json"
    
    print(f"Loading training data from: {training_file}")
    print(f"File exists: {training_file.exists()}")
    
    if not training_file.exists():
        print("Training file not found!")
        return
        
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"Training data loaded: {len(training_data['train_losses'])} epochs")
    
    # Create simple training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(training_data['train_losses']) + 1)
    
    ax.plot(epochs, training_data['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, training_data['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_file = output_dir / 'debug_training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {output_file}")
    print(f"File exists after save: {output_file.exists()}")
    
    if output_file.exists():
        print(f"File size: {output_file.stat().st_size} bytes")

if __name__ == "__main__":
    debug_visualization()