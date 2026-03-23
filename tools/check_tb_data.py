
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_tensorboard_data(log_dir):
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None

    print(f"Found {len(event_files)} event files.")
    
    # Just take the first one found for demonstration, or loop if needed
    # Usually the largest one contains the main run
    event_file = max(event_files, key=os.path.getsize)
    print(f"Processing {event_file}...")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"Available scalar tags: {tags}")

    # Extract data for a few key metrics
    data = {}
    target_metrics = ['val/rel_l2', 'val_metrics/rel_l2', 'rel_l2', 'val/psnr', 'val_metrics/psnr', 'train/loss', 'Val/RelL2', 'Val/Loss']
    
    found_metrics = []
    for tag in tags:
        for target in target_metrics:
            if target in tag:
                found_metrics.append(tag)
                
    if not found_metrics:
        print("No target metrics found in this log.")
        return None

    print(f"Extracting: {found_metrics}")
    
    for tag in found_metrics:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.DataFrame({'step': steps, 'value': values})
        print(f"  -> {tag}: {len(values)} points")
        if len(values) > 0:
            print(f"     Last value: {values[-1]}")

    return data

if __name__ == "__main__":
    # Test on a specific directory known to exist
    test_dir = "runs/AR-DR2D-Crop-Inpainting-PartialConvUNet-Size80-model_PartialConvUNet-s2025-20260216"
    extract_tensorboard_data(test_dir)
