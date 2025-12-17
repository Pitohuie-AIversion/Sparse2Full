"""
物理约束和因果性保证模块
专门为PDE时序预测设计的物理约束机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class PhysicsConstraints(nn.Module):
    """物理约束模块
    
    提供多种物理约束机制：
    1. PDE残差约束
    2. 能量守恒约束  
    3. 边界条件约束
    4. 因果性保证
    5. 物理一致性检查
    """
    
    def __init__(
        self,
        pde_type: str = 'heat',  # 'heat', 'wave', 'navier_stokes', 'reaction_diffusion'
        constraint_weights: Dict[str, float] = None,
        boundary_conditions: str = 'dirichlet',  # 'dirichlet', 'neumann', 'periodic'
        enable_causal_mask: bool = True,
        enable_energy_conservation: bool = True
    ):
        super().__init__()
        self.pde_type = pde_type
        self.boundary_conditions = boundary_conditions
        self.enable_causal_mask = enable_causal_mask
        self.enable_energy_conservation = enable_energy_conservation
        
        # 默认约束权重
        default_weights = {
            'pde_residual': 1.0,
            'energy_conservation': 0.5,
            'boundary_condition': 0.3,
            'causality': 0.2,
            'smoothness': 0.1
        }
        self.constraint_weights = constraint_weights or default_weights
        
        # PDE参数
        self.pde_params = self._init_pde_params(pde_type)
        
        # 约束网络
        self.constraint_networks = nn.ModuleDict()
        for constraint_name in self.constraint_weights.keys():
            self.constraint_networks[constraint_name] = self._build_constraint_network()
        
        # 因果性缓存
        self.causal_mask_cache = {}
        
    def _init_pde_params(self, pde_type: str) -> Dict:
        """初始化PDE参数"""
        params = {
            'heat': {
                'diffusion_coeff': 0.1,
                'thermal_conductivity': 1.0
            },
            'wave': {
                'wave_speed': 1.0,
                'damping_coeff': 0.01
            },
            'navier_stokes': {
                'viscosity': 0.01,
                'density': 1.0
            },
            'reaction_diffusion': {
                'diffusion_u': 0.1,
                'diffusion_v': 0.05,
                'reaction_rate': 1.0
            }
        }
        return params.get(pde_type, params['heat'])
    
    def _build_constraint_network(self) -> nn.Module:
        """构建约束网络"""
        return nn.Sequential(
            nn.Linear(64, 128),  # 假设输入特征维度为64
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出约束强度
        )
    
    def compute_pde_residual(self, x: torch.Tensor, dt: float = 1.0, dx: float = 1.0) -> torch.Tensor:
        """计算PDE残差"""
        if self.pde_type == 'heat':
            return self._heat_equation_residual(x, dt, dx)
        elif self.pde_type == 'wave':
            return self._wave_equation_residual(x, dt, dx)
        elif self.pde_type == 'navier_stokes':
            return self._navier_stokes_residual(x, dt, dx)
        elif self.pde_type == 'reaction_diffusion':
            return self._reaction_diffusion_residual(x, dt, dx)
        else:
            return torch.zeros_like(x)
    
    def _heat_equation_residual(self, x: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """热方程残差: u_t = α*u_xx"""
        # 时间导数（中心差分）
        if x.size(1) >= 3:
            x_t = (x[:, 2:] - x[:, :-2]) / (2 * dt)
            # 空间二阶导数
            x_xx = (x[:, 1:-1, 2:] - 2 * x[:, 1:-1, 1:-1] + x[:, 1:-1, :-2]) / (dx ** 2)
            
            residual = x_t - self.pde_params['diffusion_coeff'] * x_xx
            return F.pad(residual, (1, 1, 1, 1), mode='replicate')
        else:
            return torch.zeros_like(x)
    
    def _wave_equation_residual(self, x: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """波动方程残差: u_tt = c²*u_xx - γ*u_t"""
        if x.size(1) >= 3:
            # 时间二阶导数
            x_tt = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt ** 2)
            # 时间一阶导数
            x_t = (x[:, 2:] - x[:, :-2]) / (2 * dt)
            # 空间二阶导数
            x_xx = (x[:, 1:-1, 2:] - 2 * x[:, 1:-1, 1:-1] + x[:, 1:-1, :-2]) / (dx ** 2)
            
            wave_speed_sq = self.pde_params['wave_speed'] ** 2
            damping_coeff = self.pde_params['damping_coeff']
            
            residual = x_tt - wave_speed_sq * x_xx + damping_coeff * x_t
            return F.pad(residual, (1, 1, 1, 1), mode='replicate')
        else:
            return torch.zeros_like(x)
    
    def _navier_stokes_residual(self, x: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """Navier-Stokes方程残差（简化版本）"""
        # 这里实现一个简化的NS方程残差计算
        # 实际应用需要根据具体维度和变量数量调整
        if x.size(1) >= 3:
            viscosity = self.pde_params['viscosity']
            
            # 对流项（简化）
            convection = (x[:, 2:] - x[:, :-2]) / (2 * dt)
            
            # 扩散项（简化）
            diffusion = viscosity * (x[:, 1:-1, 2:] - 2 * x[:, 1:-1, 1:-1] + x[:, 1:-1, :-2]) / (dx ** 2)
            
            residual = convection - diffusion
            return F.pad(residual, (1, 1, 1, 1), mode='replicate')
        else:
            return torch.zeros_like(x)
    
    def _reaction_diffusion_residual(self, x: torch.Tensor, dt: float, dx: float) -> torch.Tensor:
        """反应扩散方程残差"""
        # 假设x包含两个分量u和v
        if x.size(-1) >= 2 and x.size(1) >= 3:
            u, v = x[..., 0:1], x[..., 1:2]
            
            # 扩散项
            u_xx = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)
            v_xx = (v[:, 1:-1, 2:] - 2 * v[:, 1:-1, 1:-1] + v[:, 1:-1, :-2]) / (dx ** 2)
            
            # 时间导数
            u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
            v_t = (v[:, 2:] - v[:, :-2]) / (2 * dt)
            
            # 反应项（Gray-Scott模型）
            reaction_rate = self.pde_params['reaction_rate']
            reaction_u = -u[:, 1:-1, 1:-1] * v[:, 1:-1, 1:-1] ** 2 + reaction_rate * (1 - u[:, 1:-1, 1:-1])
            reaction_v = u[:, 1:-1, 1:-1] * v[:, 1:-1, 1:-1] ** 2 - reaction_rate * v[:, 1:-1, 1:-1]
            
            # 残差
            residual_u = u_t - self.pde_params['diffusion_u'] * u_xx - reaction_u
            residual_v = v_t - self.pde_params['diffusion_v'] * v_xx - reaction_v
            
            residual = torch.cat([residual_u, residual_v], dim=-1)
            return F.pad(residual, (0, 0, 1, 1, 1, 1), mode='replicate')
        else:
            return torch.zeros_like(x)
    
    def compute_energy_conservation(self, x: torch.Tensor) -> torch.Tensor:
        """计算能量守恒约束"""
        if not self.enable_energy_conservation:
            return torch.tensor(0.0, device=x.device)
        
        # 计算总能量（L2范数）
        energy_t = torch.sum(x ** 2, dim=list(range(2, x.dim())))  # [B, T]
        
        # 能量变化率
        if x.size(1) >= 2:
            energy_change = energy_t[:, 1:] - energy_t[:, :-1]
            # 理想情况下能量变化应该接近0（守恒）
            energy_violation = torch.mean(torch.abs(energy_change))
        else:
            energy_violation = torch.tensor(0.0, device=x.device)
        
        return energy_violation
    
    def compute_boundary_condition_loss(self, x: torch.Tensor, 
                                      boundary_values: Optional[Dict] = None) -> torch.Tensor:
        """计算边界条件约束损失"""
        if self.boundary_conditions == 'periodic':
            return self._periodic_boundary_loss(x)
        elif self.boundary_conditions == 'dirichlet':
            return self._dirichlet_boundary_loss(x, boundary_values)
        elif self.boundary_conditions == 'neumann':
            return self._neumann_boundary_loss(x)
        else:
            return torch.tensor(0.0, device=x.device)
    
    def _periodic_boundary_loss(self, x: torch.Tensor) -> torch.Tensor:
        """周期性边界条件损失"""
        # 检查边界值是否相等
        left_boundary = x[..., 0]
        right_boundary = x[..., -1]
        periodic_loss = torch.mean((left_boundary - right_boundary) ** 2)
        return periodic_loss
    
    def _dirichlet_boundary_loss(self, x: torch.Tensor, 
                                boundary_values: Optional[Dict] = None) -> torch.Tensor:
        """Dirichlet边界条件损失"""
        if boundary_values is None:
            # 假设边界值为0
            boundary_loss = torch.mean(x[..., 0] ** 2) + torch.mean(x[..., -1] ** 2)
        else:
            # 使用给定的边界值
            left_val = boundary_values.get('left', 0)
            right_val = boundary_values.get('right', 0)
            left_loss = torch.mean((x[..., 0] - left_val) ** 2)
            right_loss = torch.mean((x[..., -1] - right_val) ** 2)
            boundary_loss = left_loss + right_loss
        return boundary_loss
    
    def _neumann_boundary_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Neumann边界条件损失（导数为0）"""
        # 计算边界导数
        left_derivative = x[..., 1] - x[..., 0]
        right_derivative = x[..., -1] - x[..., -2]
        neumann_loss = torch.mean(left_derivative ** 2) + torch.mean(right_derivative ** 2)
        return neumann_loss
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """获取因果掩码"""
        if not self.enable_causal_mask:
            return torch.zeros(seq_len, seq_len, device=device)
        
        cache_key = f"causal_{seq_len}"
        if cache_key not in self.causal_mask_cache:
            # 创建因果掩码（上三角为-inf）
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.bool()
            self.causal_mask_cache[cache_key] = mask
        
        return self.causal_mask_cache[cache_key]
    
    def compute_smoothness_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """计算平滑性约束"""
        # 计算二阶导数（平滑性度量）
        if x.dim() >= 3:
            # 时间平滑性
            if x.size(1) >= 3:
                second_derivative_t = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
                smoothness_t = torch.mean(second_derivative_t ** 2)
            else:
                smoothness_t = torch.tensor(0.0, device=x.device)
            
            # 空间平滑性
            if x.size(-1) >= 3:
                second_derivative_s = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
                smoothness_s = torch.mean(second_derivative_s ** 2)
            else:
                smoothness_s = torch.tensor(0.0, device=x.device)
            
            smoothness_loss = smoothness_t + smoothness_s
        else:
            smoothness_loss = torch.tensor(0.0, device=x.device)
        
        return smoothness_loss
    
    def forward(self, x: torch.Tensor, 
                physical_info: Optional[Dict] = None,
                boundary_values: Optional[Dict] = None,
                dt: float = 1.0, dx: float = 1.0) -> Dict[str, torch.Tensor]:
        """前向传播，计算所有约束损失
        
        Args:
            x: 输入张量 [B, T, C] 或 [B, T, H, W]
            physical_info: 物理信息字典
            boundary_values: 边界值字典
            dt: 时间步长
            dx: 空间步长
            
        Returns:
            约束损失字典
        """
        constraint_losses = {}
        
        # PDE残差约束
        if 'pde_residual' in self.constraint_weights:
            pde_residual = self.compute_pde_residual(x, dt, dx)
            pde_loss = torch.mean(pde_residual ** 2)
            constraint_losses['pde_residual'] = self.constraint_weights['pde_residual'] * pde_loss
        
        # 能量守恒约束
        if 'energy_conservation' in self.constraint_weights:
            energy_loss = self.compute_energy_conservation(x)
            constraint_losses['energy_conservation'] = self.constraint_weights['energy_conservation'] * energy_loss
        
        # 边界条件约束
        if 'boundary_condition' in self.constraint_weights:
            boundary_loss = self.compute_boundary_condition_loss(x, boundary_values)
            constraint_losses['boundary_condition'] = self.constraint_weights['boundary_condition'] * boundary_loss
        
        # 平滑性约束
        if 'smoothness' in self.constraint_weights:
            smoothness_loss = self.compute_smoothness_constraint(x)
            constraint_losses['smoothness'] = self.constraint_weights['smoothness'] * smoothness_loss
        
        # 总约束损失
        total_constraint_loss = sum(constraint_losses.values())
        constraint_losses['total'] = total_constraint_loss
        
        return constraint_losses


class CausalConv1d(nn.Module):
    """因果卷积，确保时间因果性"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # 计算填充，确保因果性
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation,
            groups=groups, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        conv_out = self.conv(x)
        # 移除未来时间步的影响
        if self.padding != 0:
            conv_out = conv_out[:, :, :-self.padding]
        return conv_out


class PhysicsConsistencyChecker:
    """物理一致性检查器
    
    验证预测结果是否符合基本物理定律
    """
    
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.checks = {
            'energy_conservation': self.check_energy_conservation,
            'causality': self.check_causality,
            'boundary_conditions': self.check_boundary_conditions,
            'smoothness': self.check_smoothness
        }
    
    def check_energy_conservation(self, x: torch.Tensor, threshold: float = 0.1) -> bool:
        """检查能量守恒"""
        energy_t = torch.sum(x ** 2, dim=list(range(2, x.dim())))
        if x.size(1) >= 2:
            energy_change = torch.abs(energy_t[:, 1:] - energy_t[:, :-1])
            max_change = torch.max(energy_change)
            return max_change < threshold
        return True
    
    def check_causality(self, attention_weights: torch.Tensor) -> bool:
        """检查因果性"""
        # 检查注意力权重是否违反因果性（未来影响过去）
        if attention_weights.dim() >= 3:
            B, T, _ = attention_weights.shape[:3]
            for t in range(T):
                future_influence = attention_weights[:, t, t+1:].sum()
                if future_influence > self.tolerance:
                    return False
        return True
    
    def check_boundary_conditions(self, x: torch.Tensor, 
                                expected_bc: Optional[Dict] = None) -> bool:
        """检查边界条件"""
        if expected_bc is None:
            return True
        
        # 简单的边界值检查
        left_boundary = x[..., 0]
        right_boundary = x[..., -1]
        
        if 'left' in expected_bc:
            left_diff = torch.abs(left_boundary - expected_bc['left'])
            if torch.max(left_diff) > self.tolerance:
                return False
        
        if 'right' in expected_bc:
            right_diff = torch.abs(right_boundary - expected_bc['right'])
            if torch.max(right_diff) > self.tolerance:
                return False
        
        return True
    
    def check_smoothness(self, x: torch.Tensor, threshold: float = 10.0) -> bool:
        """检查平滑性"""
        if x.size(-1) >= 3:
            # 计算二阶导数
            second_derivative = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
            smoothness_measure = torch.max(torch.abs(second_derivative))
            return smoothness_measure < threshold
        return True
    
    def comprehensive_check(self, x: torch.Tensor, 
                          attention_weights: Optional[torch.Tensor] = None,
                          expected_bc: Optional[Dict] = None) -> Dict[str, bool]:
        """综合物理一致性检查"""
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                if check_name == 'causality' and attention_weights is not None:
                    results[check_name] = check_func(attention_weights)
                elif check_name == 'boundary_conditions':
                    results[check_name] = check_func(x, expected_bc)
                else:
                    results[check_name] = check_func(x)
            except Exception as e:
                results[check_name] = False
                print(f"Check {check_name} failed with error: {e}")
        
        return results