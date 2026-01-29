# Model Audit Checklist

Based on Project Rules (v1.0)

## 1. Interface & Structure
- [ ] **Signature**: `__init__(self, in_ch, out_ch, img_size, **kwargs)`
- [ ] **Forward**: `forward(x: torch.Tensor) -> torch.Tensor` where `x` is `[B, C, H, W]`
- [ ] **Return**: Output `y` must have shape `[B, C_out, H, W]` (or scaled H, W for SR)
- [ ] **Config**: Hyperparameters (depth, heads, etc.) are passed via config, not hardcoded.

## 2. Resources & Efficiency
- [ ] **Parameters**: Total parameters ≤ 10M (Strict limit).
- [ ] **FLOPs**: Verify computational cost is reasonable for the task.
- [ ] **Memory**: Ensure no unnecessary intermediate tensors are stored.

## 3. Correctness
- [ ] **Device**: Uses `x.device` or `module.device` instead of hardcoded `cuda`.
- [ ] **Dtypes**: Compatible with AMP (Avoid explicit `.float()` unless necessary).
- [ ] **Deterministic**: No random operations without seed control (use `torch.default_generator` if needed).

## 4. Documentation
- [ ] **Docstring**: Class docstring describes architecture and input/output.
- [ ] **Type Hints**: Arguments and return values are typed.
