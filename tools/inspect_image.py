from PIL import Image
import numpy as np

img = Image.open('/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse2Full/runs_drd/AR-DR2D-EDSR-SRx4-10M-300ep-model_EDSR-s2025-20260103/test_visualizations/visualizations/predictions/sample_0203_obs_gt_pred_error_t20.png')
print(img.size)
# maybe we can just load the raw npz files if they exist?
