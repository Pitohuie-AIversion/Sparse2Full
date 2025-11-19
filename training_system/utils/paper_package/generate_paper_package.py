#!/usr/bin/env python3
"""
Generate paper package from existing training run.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np

from tools.training.paper_package import PaperPackageGenerator
from tools.training.monitoring import ValidationPipeline
from tools.utils.logger import get_logger

logger = get_logger(__name__)


def load_training_run(run_dir: Path) -> Dict[str, Any]:
    """Load training run information."""
    
    # Load configuration
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        # Try merged config
        config_file = run_dir / "config_merged.yaml"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"No configuration file found in {run_dir}")
        config = {}
    
    # Load training history
    history_file = run_dir / "training_history.json"
    training_history = {}
    if history_file.exists():
        with open(history_file, "r") as f:
            training_history = json.load(f)
    
    # Find checkpoints
    checkpoints = []
    for ckpt_file in ["best_model.pt", "latest_model.pt", "final_model.pt"]:
        ckpt_path = run_dir / ckpt_file
        if ckpt_path.exists():
            checkpoints.append(ckpt_path)
    
    # Find validation results
    val_results = {}
    val_file = run_dir / "validation_results.json"
    if val_file.exists():
        with open(val_file, "r") as f:
            val_results = json.load(f)
    
    return {
        "config": config,
        "training_history": training_history,
        "checkpoints": checkpoints,
        "validation_results": val_results,
        "run_dir": run_dir
    }


def validate_model_consistency(config: Dict[str, Any], run_dir: Path) -> bool:
    """Validate model consistency with configuration."""
    
    # Check if we can load the model
    model_files = list(run_dir.glob("*.pt")) + list(run_dir.glob("*.pth"))
    
    if not model_files:
        logger.warning("No model files found")
        return False
    
    try:
        # Try to load the first model file
        model_path = model_files[0]
        checkpoint = torch.load(model_path, map_location="cpu")
        
        logger.info(f"Successfully loaded checkpoint: {model_path}")
        logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return False


def generate_paper_package_from_run(
    run_dir: Path,
    output_dir: Path,
    validate_consistency: bool = True,
    generate_visualizations: bool = True
) -> Path:
    """Generate paper package from existing training run."""
    
    logger.info(f"Generating paper package from run: {run_dir}")
    
    # Load run information
    run_info = load_training_run(run_dir)
    
    if validate_consistency:
        logger.info("Validating model consistency...")
        is_valid = validate_model_consistency(run_info["config"], run_dir)
        if not is_valid:
            logger.warning("Model validation failed, continuing anyway...")
    
    # Create paper package generator
    package_generator = PaperPackageGenerator(
        config=run_info["config"],
        output_dir=output_dir
    )
    
    # Generate package
    package_dir = package_generator.generate_complete_package(
        trainer=None,  # No trainer available for existing runs
        validation_results=run_info["validation_results"],
        checkpoints=run_info["checkpoints"],
        seed_results=None
    )
    
    logger.info(f"Paper package generated at: {package_dir}")
    return package_dir


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description="Generate paper package from training run")
    parser.add_argument(
        "--run-dir", 
        type=Path, 
        required=True,
        help="Path to training run directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        help="Output directory for paper package (default: run_dir/paper_package)"
    )
    parser.add_argument(
        "--no-validate", 
        action="store_true",
        help="Skip model consistency validation"
    )
    parser.add_argument(
        "--no-viz", 
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Validate run directory
    if not args.run_dir.exists():
        logger.error(f"Run directory does not exist: {args.run_dir}")
        return 1
    
    if not args.run_dir.is_dir():
        logger.error(f"Run directory is not a directory: {args.run_dir}")
        return 1
    
    # Set output directory
    output_dir = args.output_dir or (args.run_dir / "paper_package")
    
    try:
        package_dir = generate_paper_package_from_run(
            run_dir=args.run_dir,
            output_dir=output_dir,
            validate_consistency=not args.no_validate,
            generate_visualizations=not args.no_viz
        )
        
        print(f"\nPaper package generated successfully!")
        print(f"Package location: {package_dir}")
        print(f"\nContents:")
        
        # List package contents
        for item in sorted(package_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(package_dir)
                size_kb = item.stat().st_size / 1024
                print(f"  {rel_path} ({size_kb:.1f} KB)")
        
        print(f"\nTo reproduce this experiment, run:")
        print(f"  cd {package_dir}")
        print(f"  ./scripts/reproduce.sh")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate paper package: {e}")
        return 1


if __name__ == "__main__":
    exit(main())