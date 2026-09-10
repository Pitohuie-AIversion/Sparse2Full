import json
import time
import os
import re

file_swe_bicubic = "runs_bicubic_cnn_swe/test_results.json"
file_swe_rbf = "runs_rbf_cnn_swe/test_results.json"
file_drd_rbf = "runs_rbf_cnn_drd/test_results.json"
file_drd_bicubic = "runs_bicubic_cnn_drd/test_results.json"

res_swe_bicubic = "runs_bicubic_cnn_swe/model_resources.json"
res_swe_rbf = "runs_rbf_cnn_swe/model_resources.json"
res_drd_rbf = "runs_rbf_cnn_drd/model_resources.json"
res_drd_bicubic = "runs_bicubic_cnn_drd/model_resources.json"

def get_metrics(test_file, res_file):
    with open(test_file, "r") as f:
        metrics = json.load(f)["final_test_metrics"]
    with open(res_file, "r") as f:
        res = json.load(f)
    return {
        "params": f"{res['params']/1e6:.2f}",
        "flops": f"{res['flops_g']:.2f}",
        "latency": f"{res['inference_latency_ms_mean']:.2f}",
        "rel_l2": f"{metrics['rel_l2']:.4f}",
        "psnr": f"{metrics['psnr']:.2f}",
        "ssim": f"{metrics['ssim']:.4f}" if "ssim" in metrics else "-",
        "herr": f"{metrics['dc_error']:.4f}" if "dc_error" in metrics else "-"
    }

print("Waiting for Bicubic-CNN DRD to finish...")
while not os.path.exists(file_drd_bicubic):
    time.sleep(10)

m_swe_bicubic = get_metrics(file_swe_bicubic, res_swe_bicubic)
m_swe_rbf = get_metrics(file_swe_rbf, res_swe_rbf)
m_drd_rbf = get_metrics(file_drd_rbf, res_drd_rbf)
m_drd_bicubic = get_metrics(file_drd_bicubic, res_drd_bicubic)

md_file = "thesis_paper/manuscript_5_chapter/chapter4_results_verification.md"
with open(md_file, "r", encoding="utf-8") as f:
    content = f.read()

# Update Table 4-2
swe_bicubic_line = f"| Bicubic-CNN | {m_swe_bicubic['params']} | {m_swe_bicubic['flops']} | {m_swe_bicubic['latency']} | {m_swe_bicubic['rel_l2']} | {m_swe_bicubic['psnr']} | 混合插值，精度次优 |"
swe_rbf_line = f"| RBF-CNN | {m_swe_rbf['params']} | {m_swe_rbf['flops']} | {m_swe_rbf['latency']} | {m_swe_rbf['rel_l2']} | {m_swe_rbf['psnr']} | 全局核矩阵影响局部重建 |"

if "Bicubic-CNN" not in content[:content.find("表 4-4")]:
    content = content.replace(
        "| MLP-Model | 0.01 | 0.14 | **0.35** | 0.0182 | 39.52 | 极简基线 |",
        f"| MLP-Model | 0.01 | 0.14 | **0.35** | 0.0182 | 39.52 | 极简基线 |\n{swe_bicubic_line}\n{swe_rbf_line}"
    )

# Update Table 4-3
drd_bicubic_line = f"| Bicubic-CNN | {m_drd_bicubic['params']} | {m_drd_bicubic['flops']} | {m_drd_bicubic['latency']} | {m_drd_bicubic['rel_l2']} | {m_drd_bicubic['psnr']} | {m_drd_bicubic['ssim']} | {m_drd_bicubic['herr']} |"
drd_rbf_line = f"| RBF-CNN | {m_drd_rbf['params']} | {m_drd_rbf['flops']} | {m_drd_rbf['latency']} | {m_drd_rbf['rel_l2']} | {m_drd_rbf['psnr']} | {m_drd_rbf['ssim']} | {m_drd_rbf['herr']} |"

if "Bicubic-CNN" not in content[content.find("表 4-3"):]:
    old_target = "| **UNetFormer** | 25.20 | 32.67 | 0.99 | 0.9473 | 16.87 | 0.0827 | 0.0000$^{\\dagger}$ |"
    new_target = f"{old_target}\n{drd_bicubic_line}\n{drd_rbf_line}"
    content = content.replace(old_target, new_target)

with open(md_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Markdown updated successfully!")
