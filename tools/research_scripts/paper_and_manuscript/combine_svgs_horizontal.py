import svgutils.transform as sg
import sys

svgs = [
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/test_sample_2_obs_gt_pred_error_t0.svg',
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/test_sample_1_obs_gt_pred_error_t0.svg',
    'runs_darcy/AR-DarcyFlow-EDSR-SRx4-beta1.0-epoch1-model_EDSR-s2025-20260110/test_visualizations/visualizations/predictions/sample_9192_obs_gt_pred_error_t0.svg'
]

figs = [sg.fromfile(s) for s in svgs]

# Get max height and total width
w = float(figs[0].width.replace("pt", "").replace("px", ""))
h = float(figs[0].height.replace("pt", "").replace("px", ""))

margin = 20

total_w = (w + margin) * len(figs)

fig = sg.SVGFigure(f"{total_w}pt", f"{h}pt")

plots = []
for i, f in enumerate(figs):
    root = f.getroot()
    root.moveto(i * (w + margin), 0)
    plots.append(root)

fig.append(plots)
fig.save("paper_package/figs/DarcyFlow/darcy_flow_verification_horizontal.svg")
print("Horizontal SVG saved successfully.")
