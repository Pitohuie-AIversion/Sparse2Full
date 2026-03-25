"""
Improved Model Loader with enhanced compatibility for various model designs.
Addresses issues found in the models directory analysis.
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


class ImprovedModelLoader:
    """
    Enhanced model loader that handles non-standard model interfaces,
    utility classes, and various parameter naming conventions.
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
    }

    # Parameter name mappings for different conventions
    PARAMETER_MAPPINGS = {
        "in_channels": [
            "in_channels",
            "in_ch",
            "in_chans",
            "n_channels",
            "input_channels",
            "input_dim",
        ],
        "out_channels": [
            "out_channels",
            "out_ch",
            "out_chans",
            "n_classes",
            "output_channels",
            "output_dim",
            "num_classes",
        ],
        "img_size": [
            "img_size",
            "image_size",
            "input_size",
            "patch_size",
            "window_size",
        ],
        "embed_dim": ["embed_dim", "hidden_dim", "d_model", "channels"],
        "depths": ["depths", "layers", "num_layers", "depth"],
        "num_heads": ["num_heads", "n_heads", "heads", "num_attention_heads"],
    }

    def __init__(self, models_dir: str = None):
        """
        Initialize the improved model loader.

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

        for filename in os.listdir(self.models_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                try:
                    self._process_module(module_name)
                except Exception as e:
                    logger.warning(f"Failed to process module {module_name}: {e}")

    def _process_module(self, module_name: str):
        """Process a single module and categorize its classes."""
        try:
            # Try different import strategies
            module = None
            import_errors = []

            # Strategy 1: Direct import
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(self.models_dir, f"{module_name}.py")
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

        # Get constructor signature
        try:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())

            # Remove 'self' from parameters
            if "self" in params:
                params.remove("self")

            # Check if it has any of the standard model parameters
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
                ]
            )

            # If it has **kwargs, it's likely a flexible model
            has_kwargs = any(
                sig.parameters[param].kind == inspect.Parameter.VAR_KEYWORD
                for param in params
            )

            # If it has both standard params and kwargs, or just standard params, it's valid
            if has_standard_params or (has_kwargs and len(params) >= 2):
                return True

            # If it only has kwargs but no other parameters, likely a utility
            if has_kwargs and len(params) == 1:
                return False

            # If it has very few parameters, likely a utility
            if len(params) < 2:
                return False

            return True

        except Exception as e:
            logger.warning(f"Error checking class {cls.__name__}: {e}")
            return False

    def _map_parameter_names(
        self, params: dict[str, Any], target_names: list[str]
    ) -> dict[str, Any]:
        """
        Map parameter names to target names using intelligent mapping.

        Args:
            params: Input parameters
            target_names: Target parameter names to map to

        Returns:
            Mapped parameters
        """
        mapped_params = {}
        used_params = set()

        for target_name in target_names:
            # Get possible source names for this target
            source_names = self.PARAMETER_MAPPINGS.get(target_name, [target_name])

            # Try to find a match in the input parameters
            for source_name in source_names:
                if source_name in params and source_name not in used_params:
                    mapped_params[target_name] = params[source_name]
                    used_params.add(source_name)
                    break

        # Add any remaining parameters that weren't mapped
        for param_name, param_value in params.items():
            if param_name not in used_params:
                mapped_params[param_name] = param_value

        return mapped_params

    def _convert_parameter_types(
        self, params: dict[str, Any], target_class: type[nn.Module]
    ) -> dict[str, Any]:
        """
        Convert parameter types based on the target class constructor.

        Args:
            params: Input parameters
            target_class: Target model class

        Returns:
            Parameters with converted types
        """
        try:
            sig = inspect.signature(target_class.__init__)
            converted_params = {}

            for param_name, param_value in params.items():
                if param_name in sig.parameters:
                    param_info = sig.parameters[param_name]

                    # Convert based on annotation if available
                    if param_info.annotation != inspect.Parameter.empty:
                        converted_params[param_name] = self._convert_type(
                            param_value, param_info.annotation
                        )
                    else:
                        converted_params[param_name] = param_value
                else:
                    # Parameter not in signature, but might be handled by **kwargs
                    converted_params[param_name] = param_value

            return converted_params

        except Exception as e:
            logger.warning(
                f"Error converting parameter types for {target_class.__name__}: {e}"
            )
            return params

    def _convert_type(self, value: Any, target_type: type) -> Any:
        """Convert a value to the target type."""
        try:
            if target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
            elif target_type == bool:
                return bool(value)
            elif target_type == str:
                return str(value)
            elif target_type == list or (
                hasattr(target_type, "__origin__") and target_type.__origin__ == list
            ):
                if isinstance(value, str):
                    # Try to parse as list
                    if "," in value:
                        return [
                            int(x.strip())
                            for x in value.split(",")
                            if x.strip().isdigit()
                        ]
                    elif value.startswith("[") and value.endswith("]"):
                        import ast

                        return ast.literal_eval(value)
                return list(value) if not isinstance(value, list) else value
            else:
                return value
        except (ValueError, TypeError, SyntaxError):
            return value

    def create_model(
        self, model_name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> nn.Module:
        """
        Create a model instance with improved compatibility.

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
        model_name = model_name.replace("_", "").lower()

        # Find matching model (case-insensitive)
        matching_models = []
        for registered_name, model_class in self.model_registry.items():
            if registered_name.lower().replace("_", "") == model_name:
                matching_models.append((registered_name, model_class))

        if not matching_models:
            available_models = list(self.model_registry.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )

        # Use the first match (should be unique)
        actual_name, model_class = matching_models[0]

        # Merge configuration and kwargs
        all_params = {}
        if config:
            # Handle nested config structures
            if "model" in config:
                model_config = config["model"]
                if isinstance(model_config, dict):
                    all_params.update(model_config)

            # Also add top-level config parameters
            for key, value in config.items():
                if key != "model" and not isinstance(value, dict):
                    all_params[key] = value

        all_params.update(kwargs)

        # Set default values for common parameters
        defaults = {
            "in_channels": 3,
            "out_channels": 3,
            "img_size": 224,
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
        }

        # Apply defaults for missing parameters
        for key, default_value in defaults.items():
            if key not in all_params:
                # Check if any mapped version exists
                mapped_names = self.PARAMETER_MAPPINGS.get(key, [key])
                if not any(name in all_params for name in mapped_names):
                    all_params[key] = default_value

        # Map parameter names to standard conventions
        mapped_params = self._map_parameter_names(all_params, list(defaults.keys()))

        # Convert parameter types based on constructor signature
        converted_params = self._convert_parameter_types(mapped_params, model_class)

        # Get constructor signature to filter parameters
        try:
            sig = inspect.signature(model_class.__init__)
            valid_params = {}
            has_kwargs = False

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

                # Add any missing parameters with defaults if available
                for param_name in sig.parameters:
                    if param_name != "self" and param_name not in valid_params:
                        param_info = sig.parameters[param_name]
                        if param_info.default != inspect.Parameter.empty:
                            valid_params[param_name] = param_info.default

            # Create model instance
            model = model_class(**valid_params)
            logger.info(
                f"Successfully created model {actual_name} with parameters: {list(valid_params.keys())}"
            )
            return model

        except Exception as e:
            logger.error(f"Failed to create model {actual_name}: {e}")
            logger.error(f"Attempted parameters: {converted_params}")
            raise ValueError(f"Failed to create model {actual_name}: {e}")

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.model_registry.keys())

    def list_utility_classes(self) -> list[str]:
        """List all utility classes (not complete models)."""
        return list(self.utility_registry.keys())

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        model_name = model_name.replace("_", "").lower()

        for registered_name, model_class in self.model_registry.items():
            if registered_name.lower().replace("_", "") == model_name:
                try:
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
                        "has_forward": hasattr(model_class, "forward"),
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


# Global instance for convenience
_improved_loader = None


def create_improved_model(
    model_name: str, config: dict[str, Any] | None = None, **kwargs
) -> nn.Module:
    """
    Create a model using the improved loader.

    Args:
        model_name: Name of the model
        config: Configuration dictionary
        **kwargs: Additional parameters

    Returns:
        Model instance
    """
    global _improved_loader
    if _improved_loader is None:
        _improved_loader = ImprovedModelLoader()

    return _improved_loader.create_model(model_name, config, **kwargs)


def list_improved_models() -> list[str]:
    """List all available models using the improved loader."""
    global _improved_loader
    if _improved_loader is None:
        _improved_loader = ImprovedModelLoader()

    return _improved_loader.list_models()


def get_improved_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a model using the improved loader."""
    global _improved_loader
    if _improved_loader is None:
        _improved_loader = ImprovedModelLoader()

    return _improved_loader.get_model_info(model_name)


if __name__ == "__main__":
    # Test the improved loader
    logging.basicConfig(level=logging.INFO)

    loader = ImprovedModelLoader()

    print(f"Available models: {len(loader.list_models())}")
    print(f"Utility classes: {len(loader.list_utility_classes())}")

    # Test creating a few models
    test_models = ["swinunet", "unet", "fno2d"]
    for model_name in test_models:
        try:
            model = loader.create_model(model_name)
            print(f"✓ Successfully created {model_name}: {type(model).__name__}")
        except Exception as e:
            print(f"✗ Failed to create {model_name}: {e}")
