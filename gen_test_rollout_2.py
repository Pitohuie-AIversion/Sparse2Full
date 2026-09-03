from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def check_eval(path):
    ea = EventAccumulator(path)
    ea.Reload()
    tags = ea.Tags()
    print(f"Tags in {path}:", tags)
    for tag in tags.get('scalars', []):
        if 'rollout' in tag.lower():
            print("Found rollout scalar:", tag)
            vals = ea.Scalars(tag)
            print("Values:", [v.value for v in vals])

check_eval('./runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/tensorboard')
check_eval('./runs_drd_paper/AR-DR2D-UNet-SRx4-10M-300ep/tensorboard')
