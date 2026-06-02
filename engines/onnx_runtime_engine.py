import numpy as np
import onnxruntime as ort
from engines.abstract_engine import AbstractEngine


class ONNXRuntimeEngine(AbstractEngine):
    """Реализация инференса через ONNX Runtime"""

    def _init_model(self) -> None:
        self.session = ort.InferenceSession(
            path_or_bytes=self.model_path,
            providers=self.__get_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __get_providers(self) -> list[str]:
        if self.device.upper() == "CPU":
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider"]

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: x})
