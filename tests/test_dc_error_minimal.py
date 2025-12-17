import torch
import torch.nn as nn
import pytest
import logging
from utils.metrics import MetricsCalculator
from ops.degradation import apply_degradation_operator

class TestDCErrorMinimal:
    def setup_method(self):
        self.logger = logging.getLogger("TestDCError")
        self.metrics_calc = MetricsCalculator()
        self.metrics_calc.logger = self.logger
        self.h_params = {
            'task': 'SR',
            'scale': 2,
            'sigma': 1.0,
            'kernel_size': 5,
            'boundary': 'mirror',
            'downsample_interpolation': 'area'
        }
        self.B, self.C, self.H, self.W = 2, 1, 32, 32
        
        # Calculate H(gt) size (16x16)
        self.h = self.H // self.h_params['scale']
        self.w = self.W // self.h_params['scale']

    def test_dc_error_correctness(self):
        """Verify DC error is 0 when H(pred) == y, where y = H(gt)"""
        gt = torch.randn(self.B, self.C, self.H, self.W)
        pred = gt.clone()
        
        obs_data = {
            'h_params': self.h_params
        }
        
        # Calculate H(gt) and set as y (real observation)
        h_gt = apply_degradation_operator(gt, obs_data)
        obs_data['y'] = h_gt
        
        # Calculate DC error: should be ||H(pred) - y|| = ||H(gt) - H(gt)|| = 0
        dc_error = self.metrics_calc.compute_data_consistency_error(pred, obs_data, norm_stats=None)
        
        assert torch.allclose(dc_error, torch.zeros_like(dc_error), atol=1e-6)

    def test_dc_error_missing_y_raises(self):
        """Verify raising error if y is missing"""
        gt = torch.randn(self.B, self.C, self.H, self.W)
        pred = gt.clone()
        obs_data = {
            'h_params': self.h_params, 
            # Missing 'y'
        }
        
        with pytest.raises(ValueError, match="obs_data must contain 'y'"):
            self.metrics_calc.compute_data_consistency_error(pred, obs_data, norm_stats=None)

    def test_dc_error_shape_mismatch_raises(self):
        """Verify strict shape checking in compute_data_consistency_error"""
        gt = torch.randn(self.B, self.C, self.H, self.W)
        pred = gt.clone()
        
        # Create a 'y' with wrong shape (e.g. 8x8 instead of 16x16)
        y_wrong = torch.randn(self.B, self.C, self.h//2, self.w//2)
        
        obs_data = {
            'h_params': self.h_params,
            'y': y_wrong
        }
        
        # Should raise because H(pred) (16x16) != y_wrong (8x8)
        # Note: apply_degradation_operator will catch this first if 'y' is in obs_data
        with pytest.raises(ValueError, match="Degradation Operator Validation Failed: Shape mismatch"):
            self.metrics_calc.compute_data_consistency_error(pred, obs_data, norm_stats=None)

    def test_degradation_shape_validation(self):
        """Verify strict shape checking in apply_degradation_operator"""
        gt = torch.randn(self.B, self.C, self.H, self.W)
        
        # Create a 'y' with wrong shape
        y_wrong = torch.randn(self.B, self.C, self.h//2, self.w//2)
        
        obs_data = {
            'h_params': self.h_params,
            'y': y_wrong
        }
        
        # Should raise because H(gt) (16x16) != y_wrong (8x8)
        with pytest.raises(ValueError, match="Degradation Operator Validation Failed: Shape mismatch"):
            apply_degradation_operator(gt, obs_data)

if __name__ == "__main__":
    t = TestDCErrorMinimal()
    t.setup_method()
    t.test_dc_error_correctness()
    print("test_dc_error_correctness PASSED")
    
    try:
        t.test_dc_error_missing_y_raises()
        print("test_dc_error_missing_y_raises PASSED")
    except Exception as e:
        print(f"FAILED test_dc_error_missing_y_raises: {e}")

    try:
        t.test_dc_error_shape_mismatch_raises()
        print("test_dc_error_shape_mismatch_raises PASSED")
    except Exception as e:
        print(f"FAILED test_dc_error_shape_mismatch_raises: {e}")
        
    try:
        t.test_degradation_shape_validation()
        print("test_degradation_shape_validation PASSED")
    except Exception as e:
        print(f"FAILED test_degradation_shape_validation: {e}")
