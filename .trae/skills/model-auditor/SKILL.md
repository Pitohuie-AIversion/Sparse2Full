---
name: model-auditor
description: Audits PyTorch model code against project standards (interface, resources, structure). Use when checking a new model implementation or reviewing code changes.
---

# Model Auditor Skill

This skill helps you audit PyTorch model implementations to ensure they comply with the Sparse2Full project rules.

## Usage

### 1. Automated Verification

Run the verification script to check interface compliance and resource usage:

```bash
python .trae/skills/model-auditor/scripts/verify_model.py <path_to_model_file.py> [--name ModelClassName]
```

This script checks:
- `__init__` signature (`in_ch`, `out_ch`, `img_size`)
- Forward pass success
- Parameter count (Limit: 10M)
- FLOPs (if `thop` is installed)
- Output shape consistency

### 2. Manual/Static Review

Use the checklist to review code structure and logic.

**Read the Checklist:** [Audit Checklist](references/audit_checklist.md)

**Key Checks:**
- **Hardcoding**: Are kernel sizes or channel numbers hardcoded? (Should be in config/init args)
- **Device Safety**: Does the code use `.cuda()` or `device='cuda'`? (Should use input tensor's device)
- **Typing**: Are type hints used?

## Example Output

```text
Checking interface for UNetFormer...
✅ __init__ signature correct.
ℹ️  Parameters: 4.20M
✅ Parameters within limit (<=10M).
✅ Forward pass successful.
✅ Output shape correct.
ℹ️  FLOPs: 1.50G
```
