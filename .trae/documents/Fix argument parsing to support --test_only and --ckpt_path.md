I found the reason why the script started training instead of testing: the command line arguments `--test_only` and `--ckpt_path` were not recognized by the script (it expected `--test-only` and `--ckpt`), causing them to be ignored and the script to default to training mode.

I will fix this by:
1.  Modifying `tools/training/train_real_data_ar.py` to accept `--test_only` (underscore) and `--ckpt_path` as valid aliases for the existing arguments.
2.  Running the command again to verify it correctly enters test mode.