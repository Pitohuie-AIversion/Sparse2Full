import sys

file_path = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# We need to insert the + L_spec results for both UNet and EDSR in Table 4-7.
# For UNet, + L_spec (RecSpec) is: Rel-L2=0.1736, PSNR=37.50, SSIM=0.8673, fRMSE-Low=18.91, Herr=0.0061
# For EDSR, + L_spec (RecSpec) is: Rel-L2=0.1885, PSNR=37.39, SSIM=0.8680, fRMSE-Low=23.55, Herr=0.0078

replacements = {
    "| **UNet** | MSE Only | 0.1758 | 37.47 | 0.8680 | 19.75 | 0.0063 |\n| | + $L_{dc}$ | 0.1743 | 37.48 | 0.8682 | 18.94 | 0.0060 |": "| **UNet** | MSE Only | 0.1758 | 37.47 | 0.8680 | 19.75 | 0.0063 |\n| | + $L_{spec}$ | 0.1736 | 37.50 | 0.8673 | 18.91 | 0.0061 |\n| | + $L_{dc}$ | 0.1743 | 37.48 | 0.8682 | 18.94 | 0.0060 |",
    "| **EDSR** | MSE Only | 0.3379 | 28.41 | 0.7459 | 70.74 | 0.0246 |\n| | + $L_{dc}$ | 0.0968 | 66.00 | 0.9074 | 13.20 | 0.0045 |": "| **EDSR** | MSE Only | 0.3379 | 28.41 | 0.7459 | 70.74 | 0.0246 |\n| | + $L_{spec}$ | 0.1885 | 37.39 | 0.8680 | 23.55 | 0.0078 |\n| | + $L_{dc}$ | 0.0968 | 66.00 | 0.9074 | 13.20 | 0.0045 |"
}

for old, new in replacements.items():
    if old in content:
        content = content.replace(old, new)
    else:
        print(f"Warning: Could not find:\n{old}\n")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated table successfully with L_spec data.")
