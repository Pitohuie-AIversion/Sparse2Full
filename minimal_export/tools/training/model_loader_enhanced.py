"""
Final Enhanced Model Loader with comprehensive model-specific parameter handling.
Addresses all identified failure patterns and provides intelligent model creation.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Any, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnhancedModelLoader:
    """
    Final enhanced model loader with comprehensive model-specific parameter handling.
    Handles all identified failure patterns from the detailed analysis.
    """

    # Utility classes that should not be treated as complete models
    UTILITY_CLASSES = {
        "Up",
        "Down",
        "DoubleConv",
        "BasicBlock",
        "Bottleneck",
        "PatchEmbed",
        "PatchMerging",
        "Mlp",
        "Attention",
        "Block",
        "WindowAttention",
        "DropPath",
        "LayerNorm",
        "GELU",
        "Conv2d",
        "Linear",
        "ReLU",
        "BatchNorm2d",
        "GroupNorm",
        "AdaptiveAvgPool2d",
        "MaxPool2d",
        "AvgPool2d",
        "Upsample",
        "ConvTranspose2d",
        "Sequential",
        "BasicLayer",
        "SwinTransformerBlock",
        "PatchExpanding",  # Component blocks
        "Rearrange",
        "FNOBottleneck",  # Utility operations
    }

    # Model-specific parameter configurations based on detailed analysis
    MODEL_SPECIFIC_CONFIGS = {
        # Complete architectures that work with standard parameters
        "SwinUNet": {
            "required_params": ["in_channels", "out_channels", "img_size"],
            "optional_params": {
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 8,
                "patch_size": 4,
                "mlp_ratio": 4.0,
                "qkv_bias": True,
                "drop_rate": 0.0,
                "drop_path_rate": 0.1,
            },
            "parameter_mapping": {
                "in_channels": ["in_channels", "in_ch", "n_channels"],
                "out_channels": ["out_channels", "out_ch", "n_classes"],
                "img_size": ["img_size", "image_size"],
            },
        },
        "SparseSwinUNet": {
            "required_params": ["in_channels", "out_channels", "img_size"],
            "optional_params": {
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "sparse_ratio": 0.5,
            },
            "parameter_mapping": {
                "in_channels": ["in_channels", "in_ch", "n_channels"],
                "out_channels": ["out_channels", "out_ch", "n_classes"],
                "img_size": ["img_size", "image_size"],
            },
        },
        "SparseAttentionEncoder": {
            "required_params": ["in_channels", "img_size"],
            "optional_params": {
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
            },
            "parameter_mapping": {
                "in_channels": ["in_channels", "in_ch", "n_channels"],
                "img_size": ["img_size", "image_size"],
            },
        },
        # Temporal components (need special handling)
        "TemporalConv1D": {
            "required_params": ["in_channels"],
            "optional_params": {
                "hidden_channels": 64,
                "num_layers": 4,
                "kernel_size": 3,
                "dilation_base": 2,
                "dropout": 0.1,
                "activation": "gelu",
            },
            "parameter_mapping": {
                "in_channels": ["in_channels", "in_ch", "n_channels"]
            },
            "input_type": "sequence",  # Special input handling
            "notes": "1D temporal convolution for sequences",
        },
        "TemporalEncoder": {
            "required_params": ["input_dim"],
            "optional_params": {
                "hidden_dim": 128,
                "num_conv_layers": 4,
                "kernel_size": 3,
                "dilation_base": 2,
                "dropout": 0.1,
                "use_positional_encoding": True,
            },
            "parameter_mapping": {
                "input_dim": ["input_dim", "in_channels", "embed_dim"]
            },
            "input_type": "sequence",
        },
        "TemporalTransformerEncoder": {
            "required_params": ["d_model"],
            "optional_params": {
                "nhead": 8,
                "num_layers": 2,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "causal": True,
                "max_seq_len": 64,
            },
            "parameter_mapping": {"d_model": ["d_model", "embed_dim", "hidden_dim"]},
            "input_type": "sequence",
        },
        # Decoder components (need encoder context)
        "SwinUNetDecoder": {
            "required_params": ["encoder_channels", "decoder_channels"],
            "optional_params": {
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 8,
                "skip_connections": True,
            },
            "notes": "Decoder component, needs encoder context",
        },
        "UNetDecoder": {
            "required_params": ["encoder_channels", "decoder_channels"],
            "optional_params": {
                "skip_connections": True,
                "upsampling_mode": "bilinear",
            },
            "notes": "UNet decoder component, needs encoder context",
        },
        # Attention components (single blocks)
        "MultiHeadCrossAttention": {
            "required_params": ["d_model", "nhead"],
            "optional_params": {"dropout": 0.1, "bias": True},
            "parameter_mapping": {
                "d_model": ["d_model", "embed_dim", "hidden_dim"],
                "nhead": ["nhead", "num_heads"],
            },
            "notes": "Single attention block",
        },
        # Abstract base classes
        "BaseModel": {
            "abstract": True,
            "notes": "Abstract base class, cannot be instantiated",
        },
        "DecoderBlock": {
            "required_params": ["in_channels", "out_channels"],
            "optional_params": {"skip_channels": 0, "upsampling_mode": "bilinear"},
            "notes": "Generic decoder block",
        },
    }

    # Extend known complete models to include UNet/FNO2d/MLPMixer
    MODEL_SPECIFIC_CONFIGS.update(
        {
            "UNet": {
                "required_params": ["in_channels", "out_channels", "img_size"],
                "optional_params": {"features": None, "bilinear": True},
                "parameter_mapping": {
                    "in_channels": ["in_channels", "in_ch", "n_channels"],
                    "out_channels": ["out_channels", "out_ch", "n_classes"],
                    "img_size": ["img_size", "image_size", "input_size"],
                },
                "input_type": "image",
            },
            "FNO2d": {
                "required_params": ["in_channels", "out_channels", "img_size"],
                "optional_params": {
                    "modes1": 12,
                    "modes2": 12,
                    "width": 64,
                    "n_layers": 4,
                    "activation": "gelu",
                },
                "parameter_mapping": {
                    "in_channels": ["in_channels", "in_ch", "n_channels"],
                    "out_channels": ["out_channels", "out_ch", "n_classes"],
                    "img_size": ["img_size", "image_size", "input_size"],
                },
                "input_type": "image",
            },
            "MLPMixer": {
                "required_params": ["in_channels", "out_channels", "img_size"],
                "optional_params": {
                    "embed_dim": 512,
                    "mlp_ratio": (0.5, 4.0),
                    "patch_size": 16,
                    "depth": 8,
                    "drop_rate": 0.0,
                    "drop_path_rate": 0.0,
                },
                "parameter_mapping": {
                    "in_channels": ["in_channels", "in_ch", "n_channels"],
                    "out_channels": ["out_channels", "out_ch", "n_classes"],
                    "img_size": ["img_size", "image_size", "input_size"],
                },
                "input_type": "image",
            },
        }
    )

    def __init__(self, models_dir: str = None):
        """
        Initialize the enhanced model loader.

        Args:
            models_dir: Directory containing model files. Defaults to ../../models
        """
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models"
        )
        self.model_registry = {}
        self.utility_registry = {}
        self._scan_models()

    def _scan_models(self):
        """Scan models directory and categorize classes."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        # Add models directory to Python path
        parent_dir = os.path.dirname(self.models_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        for root, _dirs, files in os.walk(self.models_dir):
            for filename in files:
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                file_path = os.path.join(root, filename)
                rel = os.path.relpath(file_path, self.models_dir)
                module_name = os.path.splitext(rel)[0].replace(os.sep, ".")
                try:
                    self._process_module(module_name, file_path=file_path)
                except Exception as e:
                    logger.warning(f"Failed to process module {module_name}: {e}")

    def _process_module(self, module_name: str, file_path: str | None = None):
        """Process a single module and categorize its classes."""
        try:
            # Try different import strategies
            module = None
            import_errors = []

            # Strategy 1: Direct import
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    file_path or os.path.join(self.models_dir, f"{module_name}.py"),
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
            except Exception as e:
                import_errors.append(f"Direct import: {e}")

            # Strategy 2: Relative import from models package
            if module is None:
                try:
                    module = importlib.import_module(f"models.{module_name}")
                except Exception as e:
                    import_errors.append(f"Package import: {e}")

            # Strategy 3: Absolute import
            if module is None:
                try:
                    module = __import__(module_name)
                except Exception as e:
                    import_errors.append(f"Absolute import: {e}")

            if module is None:
                logger.warning(
                    f"All import strategies failed for {module_name}: {import_errors}"
                )
                return

            # Process all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, nn.Module) and obj != nn.Module:
                    if name in self.UTILITY_CLASSES:
                        self.utility_registry[name] = obj
                        logger.debug(f"Registered utility class: {name}")
                    else:
                        # Check if it's a valid model (has forward method and reasonable constructor)
                        if self._is_valid_model_class(obj):
                            self.model_registry[name] = obj
                            logger.debug(f"Registered model class: {name}")
                        else:
                            self.utility_registry[name] = obj
                            logger.debug(
                                f"Registered as utility (not valid model): {name}"
                            )

        except Exception as e:
            logger.warning(f"Error processing module {module_name}: {e}")

    def _is_valid_model_class(self, cls: type[nn.Module]) -> bool:
        """
        Check if a class is a valid complete model (not a utility class).

        Args:
            cls: Model class to check

        Returns:
            True if it's a valid model class
        """
        # Must have forward method
        if not hasattr(cls, "forward"):
            return False

        # Check if it's an abstract class
        if inspect.isabstract(cls):
            return False

        # Get constructor signature
        try:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())

            # Remove 'self' from parameters
            if "self" in params:
                params.remove("self")

            # Check if it has any of the standard model parameters or is in our known configs
            class_name = cls.__name__
            if class_name in self.MODEL_SPECIFIC_CONFIGS:
                return True

            # Check for standard parameters
            has_standard_params = any(
                param in params
                for param in [
                    "in_channels",
                    "in_ch",
                    "in_chans",
                    "n_channels",
                    "input_channels",
                    "out_channels",
                    "out_ch",
                    "out_chans",
                    "n_classes",
                    "output_channels",
                    "img_size",
                    "image_size",
                    "input_size",
                    "d_model",
                    "input_dim",
                ]
            )

            # If it has standard params, it's likely a valid model
            if has_standard_params:
                return True

            # If it has very few parameters, likely a utility
            if len(params) < 2:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking class {cls.__name__}: {e}")
            return False

    def _get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get model-specific configuration."""
        # Normalize model name for lookup
        normalized_name = model_name.replace("_", "").lower()

        # Try exact match first
        if model_name in self.MODEL_SPECIFIC_CONFIGS:
            return self.MODEL_SPECIFIC_CONFIGS[model_name]

        # Try case-insensitive match
        for config_name in self.MODEL_SPECIFIC_CONFIGS:
            if config_name.lower().replace("_", "") == normalized_name:
                return self.MODEL_SPECIFIC_CONFIGS[config_name]

        # Return default configuration for unknown models
        return {
            "required_params": ["in_channels", "out_channels", "img_size"],
            "optional_params": {},
            "parameter_mapping": {
                "in_channels": ["in_channels", "in_ch", "n_channels"],
                "out_channels": ["out_channels", "out_ch", "n_classes"],
                "img_size": ["img_size", "image_size"],
            },
        }

    def _map_parameters_with_config(
        self, input_params: dict[str, Any], model_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Map input parameters using model-specific configuration."""
        mapped_params = {}

        # Get parameter mapping from config
        param_mapping = model_config.get("parameter_mapping", {})

        # Map required parameters
        for required_param in model_config.get("required_params", []):
            # Try to find a match in input parameters using mapping
            found = False
            possible_names = param_mapping.get(required_param, [required_param])

            for possible_name in possible_names:
                if possible_name in input_params:
                    mapped_params[required_param] = input_params[possible_name]
                    found = True
                    break

            # If not found, use default from optional params or skip
            if not found and required_param in model_config.get("optional_params", {}):
                mapped_params[required_param] = model_config["optional_params"][
                    required_param
                ]

        # Add optional parameters that are present in input
        for optional_param, default_value in model_config.get(
            "optional_params", {}
        ).items():
            if optional_param in input_params:
                mapped_params[optional_param] = input_params[optional_param]
            elif optional_param not in mapped_params:
                mapped_params[optional_param] = default_value

        # Add any additional parameters from input that weren't mapped
        for param_name, param_value in input_params.items():
            if param_name not in mapped_params:
                mapped_params[param_name] = param_value

        return mapped_params

    def _convert_parameter_types_advanced(
        self,
        params: dict[str, Any],
        target_class: type[nn.Module],
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Advanced parameter type conversion with model-specific handling."""
        try:
            sig = inspect.signature(target_class.__init__)
            converted_params = {}

            # Get expected parameter types from signature
            param_types = {}
            for param_name, param_info in sig.parameters.items():
                if param_name != "self":
                    param_types[param_name] = self._extract_type_from_annotation(
                        param_info.annotation
                    )

            # Convert each parameter based on expected type
            for param_name, param_value in params.items():
                if param_name in param_types and param_types[param_name] is not None:
                    converted_value = self._convert_value_to_type(
                        param_value, param_types[param_name]
                    )
                    if converted_value is not None:
                        converted_params[param_name] = converted_value
                    else:
                        converted_params[param_name] = param_value
                else:
                    # No type information, keep original
                    converted_params[param_name] = param_value

            return converted_params

        except Exception as e:
            logger.warning(
                f"Advanced type conversion failed for {target_class.__name__}: {e}"
            )
            return params

    def _extract_type_from_annotation(self, annotation: Any) -> type | None:
        """Extract type from parameter annotation."""
        if annotation == inspect.Parameter.empty:
            return None

        # Handle basic types
        if annotation in (int, float, bool, str):
            return annotation

        # Handle Optional types
        if hasattr(annotation, "__origin__"):
            if annotation.__origin__ is Union:
                # Get the first non-None type from Union
                for arg in annotation.__args__:
                    if arg is not type(None):
                        return arg

        # Handle List types
        if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            return list

        # Handle Tuple types
        if hasattr(annotation, "__origin__") and annotation.__origin__ is tuple:
            return tuple

        return None

    def _convert_value_to_type(self, value: Any, target_type: type) -> Any:
        """Convert a value to the target type with advanced handling."""
        try:
            if target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
            elif target_type == bool:
                return bool(value)
            elif target_type == str:
                return str(value)
            elif target_type == list:
                if isinstance(value, str):
                    # Try to parse as list
                    if value.startswith("[") and value.endswith("]"):
                        import ast

                        return ast.literal_eval(value)
                    elif "," in value:
                        # Try to parse as comma-separated integers
                        return [
                            int(x.strip())
                            for x in value.split(",")
                            if x.strip().isdigit()
                        ]
                return list(value) if not isinstance(value, list) else value
            elif target_type == tuple:
                if (
                    isinstance(value, str)
                    and value.startswith("(")
                    and value.endswith(")")
                ):
                    import ast

                    return ast.literal_eval(value)
                return tuple(value) if not isinstance(value, tuple) else value
            else:
                return value
        except (ValueError, TypeError, SyntaxError):
            return None

    def _handle_special_input_types(
        self,
        model: nn.Module,
        model_config: dict[str, Any],
        test_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Handle special input types for different model categories."""
        input_type = model_config.get("input_type", "image")

        if input_type == "sequence":
            # For temporal models, create sequence input
            batch_size = 1
            if test_input is not None:
                batch_size = test_input.shape[0]

            # Get sequence dimensions from model config or use defaults
            seq_length = 64  # Default sequence length
            input_dim = getattr(model, "input_dim", getattr(model, "d_model", 96))

            return torch.randn(batch_size, seq_length, input_dim)

        elif input_type == "decoder":
            # For decoder models, we need encoder features
            # This is a simplified approach - in practice would need actual encoder
            batch_size, channels, height, width = 1, 3, 224, 224
            if test_input is not None:
                batch_size, channels, height, width = test_input.shape

            return torch.randn(batch_size, channels, height // 4, width // 4)

        else:
            # Default image input
            return test_input if test_input is not None else torch.randn(1, 3, 224, 224)

    def create_model(
        self, model_name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> nn.Module:
        """
        Create a model instance with comprehensive parameter handling.

        Args:
            model_name: Name of the model class
            config: Configuration dictionary
            **kwargs: Additional parameters

        Returns:
            Model instance

        Raises:
            ValueError: If model not found or creation fails
        """
        # Normalize model name
        normalized_name = model_name.replace("_", "").lower()

        # Find matching model (case-insensitive)
        matching_models = []
        actual_name = None
        model_class = None

        for registered_name, cls in self.model_registry.items():
            if registered_name.lower().replace("_", "") == normalized_name:
                matching_models.append((registered_name, cls))
                actual_name = registered_name
                model_class = cls
                break

        if not matching_models:
            # Try fallback matching against known configs even if class name filtering excluded them
            for config_name in self.MODEL_SPECIFIC_CONFIGS.keys():
                if config_name.lower().replace("_", "") == normalized_name:
                    try:
                        # Attempt to import module dynamically from models package
                        module = importlib.import_module(f"models.{normalized_name}")
                        # Find class by config_name
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if name == config_name and issubclass(obj, nn.Module):
                                actual_name = config_name
                                model_class = obj
                                matching_models.append((actual_name, model_class))
                                break
                    except Exception:
                        pass

            if not matching_models:
                # Final fallback: use original loader to create model
                try:
                    from tools.training.model_loader import create_model_with_loader

                    logger.info(f"Fallback to original loader for model '{model_name}'")
                    model = create_model_with_loader(model_name, config, **kwargs)
                    # Attach config for forward test
                    try:
                        model._model_name = model_name
                        model._model_config = self._get_model_config(model_name)
                    except Exception:
                        pass
                    return model
                except Exception as e:
                    available_models = list(self.model_registry.keys())
                    raise ValueError(
                        f"Model '{model_name}' not found. Available models: {available_models}; fallback failed: {e}"
                    )

        # Get model-specific configuration
        model_config = self._get_model_config(actual_name)

        # Check if it's an abstract class
        if model_config.get("abstract", False):
            raise ValueError(f"Cannot instantiate abstract class {actual_name}")

        # Merge configuration and kwargs
        all_params = {}
        if config:
            # Handle different config types
            if hasattr(config, "model") and hasattr(config.model, "__dict__"):
                # OmegaConf or object-style config
                model_config_dict = {}
                for key, value in config.model.__dict__.items():
                    if not key.startswith("_"):
                        model_config_dict[key] = value
                all_params.update(model_config_dict)
            elif isinstance(config, dict):
                # Dictionary-style config
                if "model" in config and isinstance(config["model"], dict):
                    all_params.update(config["model"])
                # Also add top-level config parameters
                for key, value in config.items():
                    if key != "model" and not isinstance(value, dict):
                        all_params[key] = value
            elif hasattr(config, "__dict__"):
                # Simple object config - convert to dict
                config_dict = {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                }
                all_params.update(config_dict)

        all_params.update(kwargs)

        # Map parameters using model-specific configuration
        mapped_params = self._map_parameters_with_config(all_params, model_config)

        # Convert parameter types based on constructor signature
        converted_params = self._convert_parameter_types_advanced(
            mapped_params, model_class, model_config
        )

        # Get constructor signature to filter parameters
        try:
            sig = inspect.signature(model_class.__init__)
            valid_params = {}
            has_kwargs = False
            missing_required = []

            # Check for **kwargs
            for param_name, param_info in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_info.kind == inspect.Parameter.VAR_KEYWORD:
                    has_kwargs = True
                    break

            if has_kwargs:
                # Model accepts **kwargs, pass all parameters
                valid_params = converted_params
            else:
                # Filter only parameters that are in the constructor
                for param_name in sig.parameters:
                    if param_name != "self" and param_name in converted_params:
                        valid_params[param_name] = converted_params[param_name]

                # Check for missing required parameters
                for param_name in sig.parameters:
                    if param_name != "self" and param_name not in valid_params:
                        param_info = sig.parameters[param_name]
                        if param_info.default == inspect.Parameter.empty:
                            missing_required.append(param_name)

                # If we have missing required parameters, try to infer them
                if missing_required:
                    inferred_params = self._infer_missing_parameters(
                        missing_required, model_config, converted_params
                    )
                    valid_params.update(inferred_params)

                    # Check again for missing parameters
                    still_missing = [
                        p for p in missing_required if p not in valid_params
                    ]
                    if still_missing:
                        raise ValueError(
                            f"Missing required parameters for {actual_name}: {still_missing}"
                        )

            model = model_class(**valid_params)
            try:
                budget_cfg = None
                if config is not None:
                    if isinstance(config, dict):
                        budget_cfg = config.get("model_budget") or config.get(
                            "training", {}
                        ).get("model_budget")
                    else:
                        budget_cfg = getattr(config, "model_budget", None)
                        if budget_cfg is None and hasattr(config, "training"):
                            budget_cfg = getattr(config.training, "model_budget", None)
                if budget_cfg:
                    import time

                    import numpy as np

                    total_params = sum(p.numel() for p in model.parameters())
                    params_m = total_params / 1e6
                    img_size = (
                        valid_params.get("img_size")
                        or getattr(model, "img_size", None)
                        or 224
                    )
                    try:
                        from omegaconf import ListConfig  # type: ignore

                        if isinstance(img_size, ListConfig):
                            img_size = list(img_size)
                    except Exception:
                        pass
                    if isinstance(img_size, (list, tuple)):
                        if len(img_size) >= 2:
                            h, w = int(img_size[0]), int(img_size[1])
                        elif len(img_size) == 1:
                            h = w = int(img_size[0])
                        else:
                            h = w = 224
                    else:
                        h = w = int(img_size)
                    in_ch = int(
                        valid_params.get("in_channels")
                        or getattr(model, "in_channels", 3)
                        or 3
                    )
                    name_l = model.__class__.__name__.lower()
                    if "fno" in name_l:
                        flops_g = total_params * 2 * (h * w) / 1e9
                    elif (
                        ("transformer" in name_l)
                        or ("swin" in name_l)
                        or ("former" in name_l)
                    ):
                        flops_g = total_params * 4 * (h * w) / 1e9
                    elif "unet" in name_l:
                        flops_g = total_params * 2 * (h * w) / 1e9
                    elif "mlp" in name_l:
                        flops_g = total_params * 2 / 1e9
                    else:
                        flops_g = total_params * 2 * (h * w) / 1e9
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.eval()
                    model = model.to(device)
                    x = torch.randn(1, in_ch, h, w).to(device)
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(x)
                    if device != "cpu":
                        torch.cuda.synchronize()
                    times_ms = []
                    with torch.no_grad():
                        for _ in range(10):
                            t0 = time.time()
                            _ = model(x)
                            if device != "cpu":
                                torch.cuda.synchronize()
                            t1 = time.time()
                            times_ms.append((t1 - t0) * 1000.0)
                    latency_ms = float(np.mean(times_ms))
                    max_params_m = float(budget_cfg.get("max_params_m", float("inf")))
                    max_flops_g = float(budget_cfg.get("max_flops_g", float("inf")))
                    max_latency_ms = float(
                        budget_cfg.get("max_latency_ms", float("inf"))
                    )
                    violations = []
                    if params_m > max_params_m:
                        violations.append(
                            f"params {params_m:.2f}M > {max_params_m:.2f}M"
                        )
                    if flops_g > max_flops_g:
                        violations.append(f"FLOPs {flops_g:.2f}G > {max_flops_g:.2f}G")
                    if latency_ms > max_latency_ms:
                        violations.append(
                            f"latency {latency_ms:.2f}ms > {max_latency_ms:.2f}ms"
                        )
                    if violations:
                        raise ValueError(
                            "Model budget exceeded: " + "; ".join(violations)
                        )
            except Exception as e:
                logger.error(f"Model budget enforcement failed: {e}")
                raise
            logger.info(
                f"Successfully created model {actual_name} with parameters: {list(valid_params.keys())}"
            )

            # Store model config for later use
            model._model_config = model_config
            model._model_name = actual_name

            return model

        except Exception as e:
            logger.error(f"Failed to create model {actual_name}: {e}")
            logger.error(f"Attempted parameters: {converted_params}")
            raise ValueError(f"Failed to create model {actual_name}: {e}")

    def _infer_missing_parameters(
        self,
        missing_params: list[str],
        model_config: dict[str, Any],
        available_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Infer missing parameters based on model configuration and available parameters."""
        inferred = {}

        for missing_param in missing_params:
            # Try to infer from model-specific optional parameters
            if missing_param in model_config.get("optional_params", {}):
                inferred[missing_param] = model_config["optional_params"][missing_param]

            # Try to infer from available parameters using mapping
            elif "parameter_mapping" in model_config:
                # Reverse mapping - find if any available param maps to missing param
                for source, targets in model_config["parameter_mapping"].items():
                    if missing_param in targets and source in available_params:
                        inferred[missing_param] = available_params[source]
                        break

            # Special inference rules
            elif missing_param == "input_resolution":
                # Infer from img_size for transformer blocks
                if "img_size" in available_params:
                    img_size = available_params["img_size"]
                    if isinstance(img_size, (int, float)):
                        inferred[missing_param] = (
                            img_size // 4,
                            img_size // 4,
                        )  # Typical downsampling

            elif missing_param == "encoder_channels":
                # For decoders, infer typical encoder channel progression
                if "in_channels" in available_params:
                    in_channels = available_params["in_channels"]
                    inferred[missing_param] = [
                        in_channels * 8,
                        in_channels * 4,
                        in_channels * 2,
                        in_channels,
                    ]

            elif missing_param == "decoder_channels":
                # For decoders, infer typical decoder channel progression
                if "out_channels" in available_params:
                    out_channels = available_params["out_channels"]
                    inferred[missing_param] = [
                        out_channels * 8,
                        out_channels * 4,
                        out_channels * 2,
                        out_channels,
                    ]

            elif missing_param == "dim":
                # For transformer blocks, infer from embed_dim
                if "embed_dim" in available_params:
                    inferred[missing_param] = available_params["embed_dim"]
                elif "in_channels" in available_params:
                    inferred[missing_param] = (
                        available_params["in_channels"] * 32
                    )  # Typical expansion

            elif missing_param == "input_dim":
                # For temporal models, infer from in_channels
                if "in_channels" in available_params:
                    inferred[missing_param] = available_params["in_channels"]
                elif "embed_dim" in available_params:
                    inferred[missing_param] = available_params["embed_dim"]

            elif missing_param == "temporal_dim":
                # For temporal models, infer from available dimensions
                if "input_dim" in available_params:
                    inferred[missing_param] = available_params["input_dim"]
                elif "embed_dim" in available_params:
                    inferred[missing_param] = available_params["embed_dim"]
                else:
                    inferred[missing_param] = 64  # Default temporal dimension

            elif missing_param == "d_model":
                # For transformer models, infer from embed_dim
                if "embed_dim" in available_params:
                    inferred[missing_param] = available_params["embed_dim"]
                elif "in_channels" in available_params:
                    inferred[missing_param] = available_params["in_channels"] * 32

            elif missing_param == "nhead":
                # For transformer models, infer reasonable head count
                if "num_heads" in available_params:
                    inferred[missing_param] = (
                        available_params["num_heads"][0]
                        if isinstance(available_params["num_heads"], list)
                        else available_params["num_heads"]
                    )
                else:
                    inferred[missing_param] = 8  # Default head count

        return inferred

    def test_model_forward(
        self, model: nn.Module, input_shape: tuple | None = None
    ) -> bool:
        """
        Test if a model can perform a forward pass.

        Args:
            model: Model to test
            input_shape: Optional input shape override

        Returns:
            True if forward pass succeeds
        """
        try:
            # Get model configuration
            model_config = getattr(model, "_model_config", {})

            # Resolve input size from config if available
            img_size = None
            try:

                def _to_hw(v: Any) -> tuple[int, int] | None:
                    if v is None:
                        return None
                    try:
                        from omegaconf import ListConfig  # type: ignore

                        if isinstance(v, ListConfig):
                            v = list(v)
                    except Exception:
                        pass
                    if isinstance(v, (list, tuple)):
                        if len(v) >= 2:
                            return int(v[0]), int(v[1])
                        if len(v) == 1:
                            s = int(v[0])
                            return s, s
                        return None
                    s = int(v)
                    return s, s

                if hasattr(model, "img_size"):
                    hw = _to_hw(model.img_size)
                elif "img_size" in model_config:
                    hw = _to_hw(model_config["img_size"])
                else:
                    hw = None
                if hw is not None:
                    img_size = hw[0]
            except Exception:
                img_size = None

            # Determine device from model parameters
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            # Create appropriate input
            if input_shape is not None:
                test_input = torch.randn(*input_shape).to(device)
            else:
                # Prefer configured img_size over default 224
                if img_size is not None:
                    channels = getattr(model, "in_channels", 3)
                    test_input = torch.randn(1, channels, img_size, img_size).to(device)
                else:
                    test_input = self._handle_special_input_types(
                        model, model_config
                    ).to(device)

            # Test forward pass
            with torch.no_grad():
                output = model(test_input)

            logger.info(
                f"Model {getattr(model, '_model_name', type(model).__name__)} forward test successful: {test_input.shape} → {output.shape}"
            )
            return True

        except Exception as e:
            logger.warning(
                f"Model forward test failed for {getattr(model, '_model_name', type(model).__name__)}: {e}"
            )
            return False

    def list_models(self, category: str | None = None) -> list[str]:
        """
        List available models, optionally filtered by category.

        Args:
            category: Optional category filter ('complete', 'temporal', 'decoder', 'attention')

        Returns:
            List of model names
        """
        if category is None:
            return list(self.model_registry.keys())

        # Filter by category based on model configuration
        filtered_models = []
        for model_name in self.model_registry.keys():
            config = self._get_model_config(model_name)

            if (
                category == "complete"
                and config.get("input_type", "image") == "image"
                and "required_params" in config
            ):
                if (
                    "in_channels" in config["required_params"]
                    and "out_channels" in config["required_params"]
                ):
                    filtered_models.append(model_name)
            elif category == "temporal" and config.get("input_type") == "sequence":
                filtered_models.append(model_name)
            elif category == "decoder" and "encoder_channels" in config.get(
                "required_params", []
            ):
                filtered_models.append(model_name)
            elif category == "attention" and "nhead" in config.get(
                "required_params", []
            ):
                filtered_models.append(model_name)

        return filtered_models

    def list_utility_classes(self) -> list[str]:
        """List all utility classes (not complete models)."""
        return list(self.utility_registry.keys())

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get comprehensive information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with detailed model information
        """
        normalized_name = model_name.replace("_", "").lower()

        for registered_name, model_class in self.model_registry.items():
            if registered_name.lower().replace("_", "") == normalized_name:
                try:
                    config = self._get_model_config(registered_name)
                    sig = inspect.signature(model_class.__init__)
                    params = {
                        name: str(param)
                        for name, param in sig.parameters.items()
                        if name != "self"
                    }

                    return {
                        "name": registered_name,
                        "class": model_class,
                        "parameters": params,
                        "configuration": config,
                        "category": self._categorize_model(config),
                        "has_forward": hasattr(model_class, "forward"),
                        "is_abstract": inspect.isabstract(model_class),
                        "docstring": model_class.__doc__
                        or "No documentation available",
                    }
                except Exception as e:
                    return {
                        "name": registered_name,
                        "class": model_class,
                        "error": str(e),
                    }

        return {"error": f"Model {model_name} not found"}

    def _categorize_model(self, config: dict[str, Any]) -> str:
        """Categorize model based on configuration."""
        if config.get("abstract", False):
            return "abstract"
        elif config.get("input_type") == "sequence":
            return "temporal"
        elif "encoder_channels" in config.get("required_params", []):
            return "decoder"
        elif "nhead" in config.get("required_params", []):
            return "attention"
        elif "in_channels" in config.get(
            "required_params", []
        ) and "out_channels" in config.get("required_params", []):
            return "complete"
        else:
            return "other"


# Global instance for convenience
_enhanced_loader = None


def create_enhanced_model(
    model_name: str, config: dict[str, Any] | None = None, **kwargs
) -> nn.Module:
    """
    Create a model using the enhanced loader.

    Args:
        model_name: Name of the model
        config: Configuration dictionary
        **kwargs: Additional parameters

    Returns:
        Model instance
    """
    global _enhanced_loader
    if _enhanced_loader is None:
        _enhanced_loader = EnhancedModelLoader()

    return _enhanced_loader.create_model(model_name, config, **kwargs)


def list_enhanced_models(category: str | None = None) -> list[str]:
    """List available models using the enhanced loader."""
    global _enhanced_loader
    if _enhanced_loader is None:
        _enhanced_loader = EnhancedModelLoader()

    return _enhanced_loader.list_models(category)


def get_enhanced_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a model using the enhanced loader."""
    global _enhanced_loader
    if _enhanced_loader is None:
        _enhanced_loader = EnhancedModelLoader()

    return _enhanced_loader.get_model_info(model_name)


def test_enhanced_model(
    model_name: str, config: dict[str, Any] | None = None, **kwargs
) -> bool:
    """Test if a model can be created and perform forward pass."""
    global _enhanced_loader
    if _enhanced_loader is None:
        _enhanced_loader = EnhancedModelLoader()

    try:
        model = _enhanced_loader.create_model(model_name, config, **kwargs)
        # Derive input shape from config when possible
        img_size = None
        channels = None
        try:
            if config is not None:
                if hasattr(config, "data"):
                    img_size = getattr(config.data, "img_size", None)
                    channels = getattr(config.data, "channels", None)
                if hasattr(config, "model"):
                    img_size = img_size or getattr(config.model, "img_size", None)
                    channels = channels or getattr(config.model, "in_channels", None)
            img_size = img_size or kwargs.get("img_size", None)
            channels = channels or kwargs.get("in_channels", None)
        except Exception:
            pass
        try:
            from omegaconf import ListConfig  # type: ignore

            if isinstance(img_size, ListConfig):
                img_size = list(img_size)
        except Exception:
            pass

        if img_size is None:
            img_size = getattr(model, "img_size", 224)
        try:
            from omegaconf import ListConfig  # type: ignore

            if isinstance(img_size, ListConfig):
                img_size = list(img_size)
        except Exception:
            pass

        if isinstance(img_size, (list, tuple)):
            if len(img_size) >= 2:
                h, w = int(img_size[0]), int(img_size[1])
            elif len(img_size) == 1:
                h = w = int(img_size[0])
            else:
                h = w = 224
        else:
            h = w = int(img_size)
        if channels is None:
            channels = getattr(model, "in_channels", 3)
        input_shape = (1, int(channels), int(h), int(w))
        return _enhanced_loader.test_model_forward(model, input_shape=input_shape)
    except Exception as e:
        logger.error(f"Model test failed for {model_name}: {e}")
        return False


if __name__ == "__main__":
    # Test the enhanced loader
    logging.basicConfig(level=logging.INFO)

    loader = EnhancedModelLoader()

    print(f"Available models: {len(loader.list_models())}")
    print(f"Complete architectures: {len(loader.list_models('complete'))}")
    print(f"Temporal models: {len(loader.list_models('temporal'))}")
    print(f"Utility classes: {len(loader.list_utility_classes())}")

    # Test creating a few models
    test_models = ["SwinUNet", "SparseSwinUNet", "TemporalConv1D"]
    for model_name in test_models:
        try:
            model = loader.create_model(model_name)
            success = loader.test_model_forward(model)
            print(
                f"✓ {model_name}: Created and tested successfully ({'✓' if success else '✗'})"
            )
        except Exception as e:
            print(f"✗ {model_name}: {e}")
