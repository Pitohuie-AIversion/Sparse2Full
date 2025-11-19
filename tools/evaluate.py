"""兼容测试的评估工具导出

tests/test_e2e_training.py 期望从 `tools.evaluate` 导入：
- evaluate_model(model, input, target) 或类似函数

本模块提供一个最小评估函数：
返回基础指标（rel_l2、mae），使用 utils.metrics.compute_metrics。
"""

from typing import Dict, Any
import torch

try:
    from utils.metrics import compute_metrics
except Exception:
    # 轻量回退：仅计算MSE
    def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        diff = pred - target
        mse = torch.mean(diff**2).item()
        return {"mse": mse}


def evaluate_model(model: torch.nn.Module, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Dict[str, Any]:
    """最小评估函数，返回基础指标字典。"""
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)
    metrics = compute_metrics(pred, target_tensor)
    # 将可能的张量值转换为标量，便于断言
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, torch.Tensor):
                out[k] = float(v.mean().item())
            else:
                out[k] = float(v)
        except Exception:
            out[k] = v
    return out