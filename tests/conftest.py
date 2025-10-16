"""
PDEBench稀疏观测重建系统 - pytest配置文件

提供测试的全局配置、fixture和工具函数
"""

import pytest
import torch
import numpy as np
import warnings
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_configure(config):
    """pytest配置"""
    # 设置随机种子以确保测试的可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 如果有CUDA，设置CUDA种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 配置自定义标记
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture(scope="session")
def device():
    """全局设备fixture"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录路径"""
    return project_root


@pytest.fixture
def temp_config():
    """临时配置fixture"""
    return {
        'batch_size': 2,
        'image_size': 64,
        'channels': 3,
        'seed': 42
    }


@pytest.fixture
def sample_tensor(device):
    """生成样本tensor的fixture"""
    def _create_tensor(shape, dtype=torch.float32):
        torch.manual_seed(42)
        return torch.randn(shape, dtype=dtype, device=device)
    return _create_tensor


@pytest.fixture
def tolerance_config():
    """数值容差配置"""
    return {
        'atol': 1e-6,
        'rtol': 1e-5,
        'grad_atol': 1e-6,
        'grad_rtol': 1e-5
    }


class TestUtils:
    """测试工具类"""
    
    @staticmethod
    def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                                finite=True, non_negative=False):
        """检查tensor的基本属性"""
        assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
        
        if expected_shape is not None:
            assert tensor.shape == expected_shape, f"Shape mismatch: {tensor.shape} vs {expected_shape}"
        
        if expected_dtype is not None:
            assert tensor.dtype == expected_dtype, f"Dtype mismatch: {tensor.dtype} vs {expected_dtype}"
        
        if finite:
            assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
            assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
        
        if non_negative:
            assert (tensor >= 0).all(), "Tensor contains negative values"
    
    @staticmethod
    def assert_gradient_properties(tensor, check_exists=True, finite=True, non_zero=True):
        """检查梯度属性"""
        if check_exists:
            assert tensor.grad is not None, "Gradient should exist"
        
        if tensor.grad is not None:
            if finite:
                assert torch.isfinite(tensor.grad).all(), "Gradient contains non-finite values"
                assert not torch.isnan(tensor.grad).any(), "Gradient contains NaN values"
            
            if non_zero:
                grad_norm = torch.norm(tensor.grad)
                assert grad_norm > 1e-8, f"Gradient norm too small: {grad_norm}"
    
    @staticmethod
    def create_structured_data(batch_size, channels, height, width, device, pattern='mixed'):
        """创建具有特定结构的测试数据"""
        torch.manual_seed(42)
        
        # 基础随机数据
        data = torch.randn(batch_size, channels, height, width, device=device)
        
        # 创建坐标网格
        x = torch.linspace(-np.pi, np.pi, width, device=device)
        y = torch.linspace(-np.pi, np.pi, height, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        if pattern == 'low_freq':
            # 低频模式
            pattern_data = torch.sin(X) * torch.cos(Y)
            data[:, 0] += pattern_data.unsqueeze(0)
        
        elif pattern == 'high_freq':
            # 高频模式
            pattern_data = torch.sin(16 * X) * torch.cos(16 * Y)
            data[:, 0] += pattern_data.unsqueeze(0)
        
        elif pattern == 'mixed':
            # 混合频率模式
            low_freq = torch.sin(X) * torch.cos(Y)
            mid_freq = torch.sin(4 * X) * torch.cos(4 * Y)
            high_freq = torch.sin(16 * X) * torch.cos(16 * Y)
            
            if channels >= 1:
                data[:, 0] += low_freq.unsqueeze(0)
            if channels >= 2:
                data[:, 1] += mid_freq.unsqueeze(0)
            if channels >= 3:
                data[:, 2] += high_freq.unsqueeze(0)
        
        elif pattern == 'gaussian':
            # 高斯模式
            gaussian = torch.exp(-(X**2 + Y**2) / 2.0)
            data[:, 0] += gaussian.unsqueeze(0)
        
        elif pattern == 'constant':
            # 常数模式
            data.fill_(1.0)
        
        elif pattern == 'zero':
            # 零模式
            data.fill_(0.0)
        
        return data


@pytest.fixture
def test_utils():
    """测试工具fixture"""
    return TestUtils


@pytest.fixture
def sample_observation_data(device):
    """生成样本观测数据的fixture"""
    torch.manual_seed(42)
    
    # 创建观测数据字典
    obs_data = {
        'baseline': torch.randn(2, 1, 32, 32, device=device),  # 观测值
        'task': 'sr',
        'scale': 2,
        'sigma': 1.0,
        'kernel_size': 5,
        'boundary': 'mirror'
    }
    
    return obs_data


@pytest.fixture
def sample_normalization_stats(device):
    """生成样本归一化统计量的fixture"""
    torch.manual_seed(42)
    
    mu = torch.randn(1, device=device)  # 均值
    sigma = torch.abs(torch.randn(1, device=device)) + 0.1  # 标准差（确保为正）
    
    return mu, sigma


@pytest.fixture
def sample_tensor_multichannel(device):
    """生成多通道样本tensor的fixture"""
    torch.manual_seed(42)
    return torch.randn(2, 3, 64, 64, device=device)


@pytest.fixture
def sample_tensor_2d(device):
    """生成2D样本tensor的fixture"""
    torch.manual_seed(42)
    return torch.randn(2, 1, 64, 64, device=device)


def assert_tensor_close(actual, expected, atol=1e-6, rtol=1e-5, msg=""):
    """检查两个tensor是否接近"""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = torch.abs(actual - expected)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        raise AssertionError(
            f"Tensors not close enough. {msg}\n"
            f"Max difference: {max_diff:.8f}\n"
            f"Mean difference: {mean_diff:.8f}\n"
            f"Tolerance: atol={atol}, rtol={rtol}"
        )


def assert_tensor_shape(tensor, expected_shape, msg=""):
    """检查tensor形状"""
    if tensor.shape != expected_shape:
        raise AssertionError(
            f"Shape mismatch. {msg}\n"
            f"Expected: {expected_shape}\n"
            f"Actual: {tensor.shape}"
        )


def assert_tensor_finite(tensor, msg=""):
    """检查tensor是否有限"""
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"Tensor contains non-finite values. {msg}")
    if torch.isnan(tensor).any():
        raise AssertionError(f"Tensor contains NaN values. {msg}")


def assert_tensor_non_negative(tensor, msg=""):
    """检查tensor是否非负"""
    if not (tensor >= 0).all():
        raise AssertionError(f"Tensor contains negative values. {msg}")


# 标记定义
pytest_plugins = []


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 自动为GPU测试添加标记
    for item in items:
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # 为大型测试添加slow标记
        if "large" in item.name.lower() or "memory" in item.name.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """自动清理CUDA内存"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 跳过条件
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

skip_if_no_gpu_memory = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3,
    reason="Insufficient GPU memory (need at least 4GB)"
)