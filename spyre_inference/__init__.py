# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.metadata
import json
import os
from logging.config import dictConfig
from typing import Any

# Defer torch_spyre's autoload until we explicitly trigger it inside
# `TorchSpyreWorker.init_device`. Autoload loads `libspyre_comms.so`,
# which captures `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`LOCAL_WORLD_SIZE`
# at dlopen time — those env vars are only known per-worker, so the
# library can't load before init_device runs.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

__version__ = importlib.metadata.version("spyre_inference")


def _patch_granitemoehybrid_is_hybrid() -> None:
    """Ensure models with layer_types=['full_attention', ...] are not misclassified as hybrid."""
    try:
        from vllm.config.model import ModelConfig

        prop = getattr(ModelConfig, "is_hybrid", None)
        if isinstance(prop, property) and not getattr(
            ModelConfig, "_spyre_patched_is_hybrid", False
        ):
            orig_fget = prop.fget

            def is_hybrid_fixed(self) -> bool:
                if not self._model_info.is_hybrid:
                    return False
                layer_types = getattr(self.hf_config, "layer_types", None)
                if layer_types is not None and all(
                    layer in ("attention", "full_attention") for layer in layer_types
                ):
                    return False
                return orig_fget(self)

            ModelConfig.is_hybrid = property(is_hybrid_fixed)
            ModelConfig._spyre_patched_is_hybrid = True
    except Exception:
        pass


def register():
    """Register the Spyre platform."""
    _patch_granitemoehybrid_is_hybrid()
    return "spyre_inference.platform.TorchSpyrePlatform"


def register_ops():
    """Register the Spyre OOT custom ops and model adaptations."""
    _patch_granitemoehybrid_is_hybrid()
    from spyre_inference.custom_ops import register_all
    from spyre_inference.models import register_models

    register_all()
    register_models()


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
    from vllm.logger import DEFAULT_LOGGING_CONFIG, init_logger

    global logger
    logger = init_logger(__name__)

    config: dict[str, Any] = {}

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must already be done in vllm.logger
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "spyre_inference" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["spyre_inference"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["spyre_inference"] = DEFAULT_LOGGING_CONFIG["formatters"][
                    "vllm"
                ]

        if "spyre_inference" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "spyre_inference"
            config["handlers"]["spyre_inference"] = handler_config

        if "spyre_inference" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["spyre_inference"]
            config["loggers"]["spyre_inference"] = logger_config

    dictConfig(config)


_init_logging()
