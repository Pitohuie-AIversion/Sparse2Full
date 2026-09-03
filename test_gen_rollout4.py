from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

path = './runs_drd_paper/AR-DR2D-Stage2-VideoSwin-SRx4-model_unknown-s2025-20260116/tensorboard'
ea = EventAccumulator(path)
ea.Reload()

print(ea.Tags()['scalars'])
# 看看有没有可能包含每个时刻的信息，比如 Val/RelL2_step1 之类的
