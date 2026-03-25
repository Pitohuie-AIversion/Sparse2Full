#!/usr/bin/env python3
"""
Paper Package Generator CLI

This script wraps the core functionality of `tools.generate_paper_package` 
to provide a command-line interface for creating reproducible paper packages.

Usage:
    python scripts/generate_paper_package_cli.py --run-dir runs/my_experiment --output-dir paper_package/
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from tools.generate_paper_package import generate_package
except ImportError:
    print(
        "Error: Could not import tools.generate_paper_package. Ensure you are running from the project root or minimal_export directory."
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible paper package from a training run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., runs/exp_name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_package",
        help="Directory to save the package (default: paper_package)",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include model checkpoints (may be large)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Generating paper package from {run_dir}...")

    # Call the core function (assuming generate_package signature matches)
    # Adjust arguments based on actual implementation of generate_package
    try:
        generate_package(
            run_dir=run_dir,
            output_dir=Path(args.output_dir),
            include_checkpoints=args.include_checkpoints,
        )
        print(f"Success! Package generated at {args.output_dir}")
    except Exception as e:
        print(f"Failed to generate package: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
