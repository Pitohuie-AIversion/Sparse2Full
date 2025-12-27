import os
from pathlib import Path
import pytest

try:
    from datasets.temporal_pdebench import TemporalPDEBenchBase
except Exception:
    pytest.skip("缺少 datasets.temporal_pdebench 模块，跳过数据集加载测试", allow_module_level=True)


def _resolve_data_path() -> str | None:
    """根据环境变量解析PDEBench数据路径。

    优先使用 `PDEBENCH_DATA_PATH` 指定的单文件路径；
    其次在 `PDEBENCH_DATA_ROOT` 下尝试常见相对路径；
    若均不可用，则返回 None。
    """
    single = os.getenv("PDEBENCH_DATA_PATH")
    if single and Path(single).exists():
        return single

    root = os.getenv("PDEBENCH_DATA_ROOT")
    if root:
        candidates = [
            Path(root) / "2D/diffusion-reaction/2D_diff-react_NA_NA.h5",
            Path(root) / "diffusion-reaction/2D_diff-react_NA_NA.h5",
            Path(root) / "DR2D/2D_diff-react_NA_NA.h5",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
    return None


def test_dataset_loading_basic():
    """基础数据集加载测试：在存在数据时验证能正常构建并读取一个样本。"""
    data_path = _resolve_data_path()
    if not data_path:
        pytest.skip("缺少 Diffusion-Reaction 数据集（设置 PDEBENCH_DATA_ROOT 或 PDEBENCH_DATA_PATH）")

    dataset = TemporalPDEBenchBase(
        data_path=data_path,
        keys=['0000'],  # 使用一个通道/时间键以降低依赖
        T_in=2,
        T_out=2,
        dt=1,
        normalize=True,
        use_official_format=True,
        image_size=(64, 64),
    )

    assert len(dataset) > 0, "数据集不应为空"
    sample = dataset[0]
    assert sample is not None, "应能读取到一个样本"