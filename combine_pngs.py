from PIL import Image

pngs = [
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/test_sample_2_obs_gt_pred_error_t0.png',
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0.png',
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/sample_9192_obs_gt_pred_error_t0.png'
]

images = [Image.open(p) for p in pngs]
widths, heights = zip(*(i.size for i in images))

max_width = max(widths)
total_height = sum(heights)

new_im = Image.new('RGB', (max_width, total_height), (255, 255, 255))

y_offset = 0
for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

new_im.save('paper_package/figs/DarcyFlow/darcy_flow_verification_combined.png')
print("Combined PNG saved successfully.")
