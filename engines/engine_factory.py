from typing import Any
from engines.abstract_engine import AbstractEngine
from engines.open_vino_engine import OpenVINOEngine
from engines.onnx_runtime_engine import ONNXRuntimeEngine


class EngineFactory:
    @staticmethod
    def get_engine(
        config: dict[str, Any],
        model_path: str,
        device: str,
        **kwargs: Any
    ) -> AbstractEngine:
        """Фабричный метод выбора бэкенда инференса"""

        backend = config.get("backend", "openvino").lower()

        if backend == "openvino":
            return OpenVINOEngine(model_path, device, **kwargs)
        elif backend == "onnxruntime":
            return ONNXRuntimeEngine(model_path, device)
        else:
            raise ValueError(f"Неизвестный бэкенд: {backend}")
