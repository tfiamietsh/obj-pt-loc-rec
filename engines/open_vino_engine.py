import numpy as np
import openvino as ov
from engines.abstract_engine import AbstractEngine


class OpenVINOEngine(AbstractEngine):
    """Реализация инференса через OpenVINO"""

    def __init__(
        self,
        model_path: str,
        device: str,
        dynamic_batch: bool = False,
        target_size: tuple = None
    ) -> None:
        self.__dynamic_batch = dynamic_batch
        self.__target_size = target_size
        super().__init__(model_path, device)

    def _init_model(self) -> None:
        core = ov.Core()
        model = core.read_model(model=self.model_path)

        input_layer = model.inputs[0]
        if self.__dynamic_batch:
            model.reshape({input_layer: ov.PartialShape([-1, 3, 224, 224])})
        elif self.__target_size:
            h, w = self.__target_size

            model.reshape({input_layer: ov.PartialShape([1, 3, h, w])})

        compiled_model = core.compile_model(model, device_name=self.device)

        self.infer_request = compiled_model.create_infer_request()
        self.input_port = compiled_model.input(0)

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        outputs = self.infer_request.infer({self.input_port: x})

        return list(outputs.values())
