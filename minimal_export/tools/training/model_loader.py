"""
模型加载器 - 支持自动识别和加载models目录下的所有模型架构
提供统一的模型创建接口，兼容不同的模型配置
"""

import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ModelLoader:
    """统一的模型加载器，支持自动识别和加载models目录下的所有模型"""

    def __init__(self, models_dir: str = None):
        """
        初始化模型加载器

        Args:
            models_dir: 模型目录路径，默认为项目根目录下的models文件夹
        """
        if models_dir is None:
            # 默认路径：项目根目录/models
            project_root = Path(__file__).resolve().parents[2]
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)

        self.models_dir = Path(models_dir)
        self._available_models: dict[str, type[nn.Module]] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}
        self._scan_models()

    def _scan_models(self):
        """扫描models目录下的所有模型文件"""
        logger.info(f"扫描模型目录: {self.models_dir}")

        # 扫描models目录下的所有Python文件
        for py_file in self.models_dir.rglob("*.py"):
            if py_file.name.startswith("__") or py_file.name.startswith("."):
                continue

            try:
                # 计算相对导入路径
                relative_path = py_file.relative_to(self.models_dir.parent)
                module_path = str(relative_path.with_suffix("")).replace(os.sep, ".")

                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_path, py_file)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 查找模型类
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, nn.Module)
                        and obj != nn.Module
                        and not name.startswith("_")
                        and hasattr(obj, "__init__")
                    ):

                        # 检查是否是基础模型类（需要实例化参数）
                        try:
                            sig = inspect.signature(obj.__init__)
                            params = list(sig.parameters.keys())

                            # 检查是否有必要的初始化参数
                            if any(
                                param in params
                                for param in ["in_channels", "in_ch", "input_channels"]
                            ):
                                model_key = name.lower()
                                self._available_models[model_key] = obj

                                # 提取默认参数
                                default_params = {}
                                for param_name, param in sig.parameters.items():
                                    if param_name == "self":
                                        continue
                                    if param.default != inspect.Parameter.empty:
                                        default_params[param_name] = param.default

                                self._model_configs[model_key] = {
                                    "class_name": name,
                                    "module_path": module_path,
                                    "file_path": str(py_file),
                                    "default_params": default_params,
                                    "init_params": params,
                                }

                                logger.debug(f"发现模型: {name} -> {model_key}")

                        except Exception as e:
                            logger.debug(f"检查模型类 {name} 失败: {e}")
                            continue

            except Exception as e:
                logger.debug(f"扫描文件 {py_file} 失败: {e}")
                continue

        # 也扫描models/__init__.py中定义的模型
        try:
            import sys

            sys.path.insert(0, str(self.models_dir.parent))
            from models import create_model

            # 获取create_model支持的模型列表
            self._external_model_factory = create_model
            logger.info("已加载外部模型工厂函数")

        except Exception as e:
            logger.debug(f"加载外部模型工厂失败: {e}")
            self._external_model_factory = None

        logger.info(f"扫描完成，发现 {len(self._available_models)} 个模型类")
        for key, model_class in self._available_models.items():
            logger.info(f"  - {key}: {model_class.__name__}")

    def list_available_models(self) -> list[str]:
        """获取所有可用模型名称"""
        models = list(self._available_models.keys())
        if self._external_model_factory:
            # 这里可以添加外部工厂支持的模型名称
            models.extend(["unet", "fno2d", "swin_unet", "segformer", "hybrid"])

        # 过滤掉非独立模型的组件
        exclude_components = {
            "down",
            "up",
            "downsample",
            "upsample",
            "fourierblock2d",
            "spectralconv2d",
            "stablespectralconv2d",
            "outconv",
            "pconvdoubleconv",
            "pconvdown",
            "pconvup",
            "convbnact",
            "denselayer",
            "basetemporalmodel",
            "convlstmcell",
            "simplespatialcnn",
            "simplespatialtemporalcnn",
            "causalconv1d",
            "spatialfeatureextractor",
            "temporalconv1d",
            "physicstransformertemporal",
            "convtemporalpredictor",
            "branchencoder",
            "overlappatchembed",
            "basemodel",
            "sparseattentionencoder",
            "sparseswinunet",
            "swintemporalwrapper",
            "partialconv2d",
        }

        filtered_models = [m for m in models if m.lower() not in exclude_components]
        return sorted(list(set(filtered_models)))

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """获取模型信息"""
        model_key = model_name.lower()
        return self._model_configs.get(model_key)

    def create_model(
        self,
        model_name: str,
        config: DictConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> nn.Module:
        """
        创建模型实例 - 增强错误处理和兼容性检查

        Args:
            model_name: 模型名称
            config: 模型配置（可选）
            **kwargs: 额外的模型参数

        Returns:
            模型实例

        Raises:
            ValueError: 当模型不支持或参数错误时
            RuntimeError: 当模型创建失败时
        """
        model_key = model_name.lower()
        logger.info(f"尝试创建模型: {model_name} (key: {model_key})")

        # 首先尝试使用外部工厂函数
        if self._external_model_factory:
            try:
                logger.info(f"使用外部工厂函数创建模型: {model_name}")

                # 准备参数: 优先使用kwargs，其次从config提取
                factory_kwargs = {}

                # 1. 从config提取参数
                if config is not None:
                    if isinstance(config, DictConfig):
                        config_dict = OmegaConf.to_container(config, resolve=True)
                    else:
                        config_dict = dict(config)

                    # 尝试提取model部分或顶层参数
                    if "model" in config_dict:
                        factory_kwargs.update(config_dict["model"])
                    factory_kwargs.update(
                        {
                            k: v
                            for k, v in config_dict.items()
                            if k
                            in ["in_channels", "out_channels", "img_size", "embed_dim"]
                        }
                    )

                # 2. 合并kwargs (最高优先级)
                factory_kwargs.update(kwargs)

                # 3. 调用工厂函数 (使用 name + kwargs 方式，避免config对象解析歧义)
                model = self._external_model_factory(model_name, **factory_kwargs)
                logger.info(f"外部工厂成功创建模型: {type(model).__name__}")
                return model

            except Exception as e:
                logger.warning(f"外部工厂创建模型 {model_name} 失败: {e}")
                logger.info("尝试使用内部模型加载器...")

        # 尝试使用内部扫描的模型类
        if model_key not in self._available_models:
            available = self.list_available_models()
            error_msg = f"不支持的模型: {model_name}. 可用模型: {available}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        model_class = self._available_models[model_key]
        model_info = self._model_configs[model_key]

        logger.info(
            f"使用内部模型类: {model_info['class_name']} from {model_info['file_path']}"
        )

        # 合并配置参数
        final_params = {}

        # 1. 使用默认参数
        final_params.update(model_info["default_params"])
        logger.debug(f"默认参数: {model_info['default_params']}")

        # 2. 使用配置文件中的参数
        if config is not None:
            try:
                if isinstance(config, DictConfig):
                    config_dict = OmegaConf.to_container(config, resolve=True)
                else:
                    config_dict = dict(config)

                # 提取模型相关参数
                model_config = config_dict.get("model", {})
                if isinstance(model_config, dict):
                    final_params.update(model_config)
                    logger.debug(f"模型配置参数: {model_config}")

                # 也支持顶层配置
                for key in [
                    "in_channels",
                    "out_channels",
                    "img_size",
                    "input_channels",
                    "output_channels",
                ]:
                    if key in config_dict:
                        final_params[key] = config_dict[key]
                        logger.debug(f"顶层配置参数 {key}: {config_dict[key]}")

            except Exception as e:
                logger.warning(f"解析配置失败: {e}")

        # 3. 使用显式传入的参数（最高优先级）
        if kwargs:
            final_params.update(kwargs)
            logger.debug(f"显式参数: {kwargs}")

        # 参数映射和兼容性处理
        param_mapping = {
            "in_channels": ["in_channels", "in_ch", "input_channels", "input_ch"],
            "out_channels": ["out_channels", "out_ch", "output_channels", "output_ch"],
            "img_size": ["img_size", "image_size", "input_size"],
            "embed_dim": ["embed_dim", "embedding_dim", "hidden_dim"],
            "num_heads": ["num_heads", "n_heads", "attention_heads"],
            "depths": ["depths", "layers", "num_layers"],
            "drop_rate": ["drop_rate", "dropout", "dropout_rate"],
            "mlp_ratio": ["mlp_ratio", "mlp_expand_ratio"],
        }

        # 标准化参数名称
        normalized_params = {}
        for standard_name, aliases in param_mapping.items():
            for alias in aliases:
                if alias in final_params:
                    normalized_params[standard_name] = final_params[alias]
                    logger.debug(
                        f"参数映射: {alias} -> {standard_name} = {final_params[alias]}"
                    )
                    break

        # 保留未映射的参数
        for key, value in final_params.items():
            if not any(key in aliases for aliases in param_mapping.values()):
                normalized_params[key] = value
                logger.debug(f"保留未映射参数: {key} = {value}")

        # 必需的参数检查
        required_params = ["in_channels", "out_channels", "img_size"]
        missing_params = [p for p in required_params if p not in normalized_params]

        if missing_params:
            logger.warning(f"缺少必需参数: {missing_params}")
            # 尝试从配置中推断
            if config is not None:
                try:
                    if isinstance(config, DictConfig):
                        config_dict = OmegaConf.to_container(config, resolve=True)
                    else:
                        config_dict = dict(config)

                    # 从data配置推断
                    data_config = config_dict.get("data", {})
                    if "in_channels" in missing_params:
                        if "channels" in data_config:
                            normalized_params["in_channels"] = data_config["channels"]
                            logger.info(
                                f"从data配置推断 in_channels: {data_config['channels']}"
                            )
                        elif "input_channels" in data_config:
                            normalized_params["in_channels"] = data_config[
                                "input_channels"
                            ]
                            logger.info(
                                f"从data配置推断 in_channels: {data_config['input_channels']}"
                            )

                    if "out_channels" in missing_params:
                        if "channels" in data_config:
                            normalized_params["out_channels"] = data_config["channels"]
                            logger.info(
                                f"从data配置推断 out_channels: {data_config['channels']}"
                            )
                        elif "output_channels" in data_config:
                            normalized_params["out_channels"] = data_config[
                                "output_channels"
                            ]
                            logger.info(
                                f"从data配置推断 out_channels: {data_config['output_channels']}"
                            )

                    if "img_size" in missing_params:
                        if "img_size" in data_config:
                            normalized_params["img_size"] = data_config["img_size"]
                            logger.info(
                                f"从data配置推断 img_size: {data_config['img_size']}"
                            )
                        elif "image_size" in data_config:
                            normalized_params["img_size"] = data_config["image_size"]
                            logger.info(
                                f"从data配置推断 img_size: {data_config['image_size']}"
                            )

                except Exception as e:
                    logger.warning(f"从配置推断参数失败: {e}")

        # 最终检查必需参数
        missing_params = [p for p in required_params if p not in normalized_params]
        if missing_params:
            error_msg = f"缺少必需的模型参数: {missing_params}. 请确保配置中包含这些参数或通过kwargs传入"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 类型转换
        type_conversions = {
            "in_channels": int,
            "out_channels": int,
            "img_size": int,
            "patch_size": int,
            "embed_dim": int,
            "mlp_ratio": float,
            "drop_rate": float,
            "attn_drop_rate": float,
            "drop_path_rate": float,
        }

        for param_name, converter in type_conversions.items():
            if param_name in normalized_params:
                try:
                    original_value = normalized_params[param_name]
                    if param_name == "mlp_ratio":
                        if isinstance(original_value, (list, tuple)):
                            converted_value = original_value
                        elif isinstance(original_value, str):
                            try:
                                import ast

                                parsed = ast.literal_eval(original_value)
                                converted_value = (
                                    parsed
                                    if isinstance(parsed, (list, tuple))
                                    else float(parsed)
                                )
                            except Exception:
                                converted_value = float(original_value)
                        else:
                            converted_value = float(original_value)
                    else:
                        converted_value = converter(original_value)
                    normalized_params[param_name] = converted_value
                    logger.debug(
                        f"参数类型转换 {param_name}: {original_value} -> {converted_value}"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"参数 {param_name} 类型转换失败: {e}")

        # 特殊处理：将列表字符串转换为列表
        list_params = ["depths", "num_heads"]
        for param_name in list_params:
            if param_name in normalized_params and isinstance(
                normalized_params[param_name], str
            ):
                try:
                    # 尝试解析类似 "[2, 2, 6, 2]" 的字符串
                    import ast

                    original_value = normalized_params[param_name]
                    converted_value = ast.literal_eval(original_value)
                    normalized_params[param_name] = converted_value
                    logger.debug(
                        f"列表参数转换 {param_name}: {original_value} -> {converted_value}"
                    )
                except Exception as e:
                    logger.warning(
                        f"无法解析列表参数 {param_name}: {normalized_params[param_name]}, 错误: {e}"
                    )

        logger.info(f"创建模型 {model_name} 使用参数: {normalized_params}")

        try:
            # 获取模型类的参数签名
            import inspect

            sig = inspect.signature(model_class.__init__)
            valid_params = set(sig.parameters.keys())

            # 过滤有效参数
            filtered_params = {}
            for param_name, param_value in normalized_params.items():
                if param_name in valid_params:
                    filtered_params[param_name] = param_value
                else:
                    logger.warning(
                        f"参数 {param_name} 不在模型 {model_class.__name__} 的构造函数中，将被忽略"
                    )

            # 检查必需参数
            required_params = []
            for param_name, param in sig.parameters.items():
                if param_name != "self" and param.default == inspect.Parameter.empty:
                    # 忽略 **kwargs 参数
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        continue
                    if param_name not in filtered_params:
                        required_params.append(param_name)

            if required_params:
                # 尝试从配置中推断缺失的必需参数
                for param_name in required_params:
                    if param_name == "in_ch" and "in_channels" in filtered_params:
                        filtered_params["in_ch"] = filtered_params["in_channels"]
                        logger.info("映射参数 in_channels -> in_ch")
                    elif param_name == "out_ch" and "out_channels" in filtered_params:
                        filtered_params["out_ch"] = filtered_params["out_channels"]
                        logger.info("映射参数 out_channels -> out_ch")
                    elif (
                        param_name == "in_channel" and "in_channels" in filtered_params
                    ):
                        filtered_params["in_channel"] = filtered_params["in_channels"]
                        logger.info("映射参数 in_channels -> in_channel")
                    elif (
                        param_name == "out_channel"
                        and "out_channels" in filtered_params
                    ):
                        filtered_params["out_channel"] = filtered_params["out_channels"]
                        logger.info("映射参数 out_channels -> out_channel")
                    elif (
                        param_name == "input_channels"
                        and "in_channels" in filtered_params
                    ):
                        filtered_params["input_channels"] = filtered_params[
                            "in_channels"
                        ]
                        logger.info("映射参数 in_channels -> input_channels")
                    elif (
                        param_name == "output_channels"
                        and "out_channels" in filtered_params
                    ):
                        filtered_params["output_channels"] = filtered_params[
                            "out_channels"
                        ]
                        logger.info("映射参数 out_channels -> output_channels")
                    elif param_name == "image_size" and "img_size" in filtered_params:
                        filtered_params["image_size"] = filtered_params["img_size"]
                        logger.info("映射参数 img_size -> image_size")
                    elif param_name == "input_size" and "img_size" in filtered_params:
                        filtered_params["input_size"] = filtered_params["img_size"]
                        logger.info("映射参数 img_size -> input_size")

            # 再次检查必需参数（忽略 *args/**kwargs）
            missing_required = []
            for param_name, param in sig.parameters.items():
                if param_name != "self" and param.default == inspect.Parameter.empty:
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    if param_name not in filtered_params:
                        missing_required.append(param_name)

            if missing_required:
                error_msg = (
                    f"模型 {model_class.__name__} 缺少必需参数: {missing_required}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"最终使用参数: {filtered_params}")

            # 创建模型实例
            logger.info(f"实例化模型类: {model_info['class_name']}")
            model = model_class(**filtered_params)
            logger.info(f"成功创建模型: {model_class.__name__}")

            # 验证模型基本属性
            self._validate_model_attributes(model, normalized_params)

            return model

        except Exception as e:
            error_msg = f"创建模型 {model_name} 失败: {e}"
            logger.error(error_msg)
            logger.error(f"使用的参数: {normalized_params}")
            logger.error(f"模型类: {model_info['class_name']}")
            logger.error(f"模型文件: {model_info['file_path']}")
            raise RuntimeError(error_msg) from e

    def _validate_model_attributes(
        self, model: nn.Module, params: dict[str, Any]
    ) -> None:
        """
        验证模型基本属性

        Args:
            model: 模型实例
            params: 创建参数

        Raises:
            ValueError: 模型属性验证失败
        """
        try:
            # 检查模型是否为 nn.Module 实例
            if not isinstance(model, nn.Module):
                raise ValueError(
                    f"创建的模型不是 nn.Module 实例，实际类型: {type(model)}"
                )

            # 验证输入输出通道数
            if hasattr(model, "in_channels") and "in_channels" in params:
                model_in_channels = model.in_channels
                expected_in_channels = params["in_channels"]
                if model_in_channels != expected_in_channels:
                    logger.warning(
                        f"模型输入通道数不一致: 期望={expected_in_channels}, 实际={model_in_channels}"
                    )

            if hasattr(model, "out_channels") and "out_channels" in params:
                model_out_channels = model.out_channels
                expected_out_channels = params["out_channels"]
                if model_out_channels != expected_out_channels:
                    logger.warning(
                        f"模型输出通道数不一致: 期望={expected_out_channels}, 实际={model_out_channels}"
                    )

            if hasattr(model, "img_size") and "img_size" in params:
                model_img_size = model.img_size
                expected_img_size = params["img_size"]
                if model_img_size != expected_img_size:
                    logger.warning(
                        f"模型图像尺寸不一致: 期望={expected_img_size}, 实际={model_img_size}"
                    )

            # 检查模型是否有 forward 方法
            if not hasattr(model, "forward") or not callable(model.forward):
                raise ValueError("模型缺少 forward 方法")

            # 尝试获取模型参数数量
            try:
                param_count = sum(p.numel() for p in model.parameters())
                logger.info(f"模型参数数量: {param_count:,}")
            except Exception as e:
                logger.warning(f"无法计算模型参数数量: {e}")

            logger.info(f"模型属性验证通过: {type(model).__name__}")

        except Exception as e:
            logger.error(f"模型属性验证失败: {e}")
            raise ValueError(f"模型属性验证失败: {e}") from e

    def validate_model_compatibility(
        self, model: nn.Module, config: dict[str, Any]
    ) -> bool:
        """
        验证模型兼容性

        Args:
            model: 模型实例
            config: 配置字典

        Returns:
            是否兼容
        """
        try:
            # 检查输入输出通道
            expected_in_channels = config.get(
                "in_channels", config.get("data", {}).get("channels", 1)
            )
            expected_out_channels = config.get(
                "out_channels", config.get("data", {}).get("channels", 1)
            )
            expected_img_size = config.get(
                "img_size", config.get("data", {}).get("img_size", 128)
            )

            # 尝试获取模型属性
            model_in_channels = getattr(model, "in_channels", None)
            model_out_channels = getattr(model, "out_channels", None)
            model_img_size = getattr(model, "img_size", None)

            if (
                model_in_channels is not None
                and model_in_channels != expected_in_channels
            ):
                logger.warning(
                    f"输入通道不匹配: 模型={model_in_channels}, 期望={expected_in_channels}"
                )
                return False

            if (
                model_out_channels is not None
                and model_out_channels != expected_out_channels
            ):
                logger.warning(
                    f"输出通道不匹配: 模型={model_out_channels}, 期望={expected_out_channels}"
                )
                return False

            if model_img_size is not None and model_img_size != expected_img_size:
                logger.warning(
                    f"图像尺寸不匹配: 模型={model_img_size}, 期望={expected_img_size}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"验证模型兼容性失败: {e}")
            return False


def check_model_health(model: nn.Module, input_shape: tuple = None) -> dict[str, Any]:
    """
    检查模型健康状况

    Args:
        model: 模型实例
        input_shape: 输入形状，默认为 (1, 1, 128, 128)

    Returns:
        健康检查结果
    """
    if input_shape is None:
        input_shape = (1, 1, 128, 128)

    health_report = {
        "model_type": type(model).__name__,
        "parameters": 0,
        "trainable_parameters": 0,
        "forward_pass": False,
        "output_shape": None,
        "memory_usage_mb": 0,
        "errors": [],
    }

    try:
        # 计算参数数量
        health_report["parameters"] = sum(p.numel() for p in model.parameters())
        health_report["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # 测试前向传播
        try:
            import torch

            test_input = torch.randn(*input_shape)
            model.eval()
            with torch.no_grad():
                test_output = model(test_input)
                health_report["forward_pass"] = True
                health_report["output_shape"] = (
                    list(test_output.shape)
                    if hasattr(test_output, "shape")
                    else str(type(test_output))
                )

                # 检查输出是否包含 NaN 或无穷大
                if torch.isnan(test_output).any():
                    health_report["errors"].append("输出包含 NaN 值")
                if torch.isinf(test_output).any():
                    health_report["errors"].append("输出包含无穷大值")

        except Exception as e:
            health_report["errors"].append(f"前向传播失败: {str(e)}")

        # 估算内存使用
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            health_report["memory_usage_mb"] = param_size / (1024 * 1024)
        except Exception as e:
            health_report["errors"].append(f"内存使用估算失败: {str(e)}")

        health_report["status"] = (
            "healthy" if not health_report["errors"] else "unhealthy"
        )

    except Exception as e:
        health_report["errors"].append(f"健康检查失败: {str(e)}")
        health_report["status"] = "failed"

    return health_report


# 全局模型加载器实例
_model_loader = None


def get_model_loader(models_dir: str = None) -> ModelLoader:
    """获取全局模型加载器实例"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(models_dir)
    return _model_loader


def create_model_with_loader(
    model_name: str,
    config: DictConfig | dict[str, Any] | None = None,
    models_dir: str = None,
    **kwargs,
) -> nn.Module:
    """
    使用模型加载器创建模型实例

    Args:
        model_name: 模型名称
        config: 模型配置（可选）
        models_dir: 模型目录（可选）
        **kwargs: 额外的模型参数

    Returns:
        模型实例
    """
    loader = get_model_loader(models_dir)
    return loader.create_model(model_name, config, **kwargs)


def list_models(models_dir: str = None) -> list[str]:
    """获取所有可用模型列表"""
    loader = get_model_loader(models_dir)
    return loader.list_available_models()


def get_model_info(model_name: str, models_dir: str = None) -> dict[str, Any] | None:
    """获取模型信息"""
    loader = get_model_loader(models_dir)
    return loader.get_model_info(model_name)
