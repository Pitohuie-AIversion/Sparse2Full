import os
import glob
import shutil
from pathlib import Path
import argparse

def find_and_copy_best_result(run_dir, output_path):
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    
    # Search for combined images
    combined_files = list(run_dir.rglob('*obs_gt_pred_error*.png'))
    
    if combined_files:
        print(f"Found {len(combined_files)} combined visualization files.")
        best_file = combined_files[0] # Pick the first one as representative
        print(f"Selecting {best_file} for Chapter 4.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_file, output_path)
        print(f"Successfully copied to {output_path}")
    else:
        print("No combined visualization files found. Looking for separate ob, prd, gt, err files...")
        # Add logic for separate files if needed later
        # For now, we know the codebase generates combined ones
        print("Please ensure your evaluation script has generated the visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and arrange result figures for Chapter 4.")
    parser.add_argument("--run_dir", type=str, default="../runs", help="Base directory containing runs")
    parser.add_argument("--output", type=str, default="../thesis_paper/manuscript_5_chapter/images/fig4-1_vis_results.png", help="Output image path")
    args = parser.parse_args()
    
    find_and_copy_best_result(args.run_dir, args.output)
