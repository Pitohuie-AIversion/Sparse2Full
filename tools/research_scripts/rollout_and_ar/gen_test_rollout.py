from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def extract_from_tb(path):
    try:
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags()
        print("Tags found:", tags)
        if 'scalars' in tags and 'Val/RelL2' in tags['scalars']:
            vals = ea.Scalars('Val/RelL2')
            print(f"Found {len(vals)} RelL2 values.")
            return [v.value for v in vals]
    except Exception as e:
        print("Error:", e)
    return None

print("Checking AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116")
extract_from_tb('./runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/tensorboard')
