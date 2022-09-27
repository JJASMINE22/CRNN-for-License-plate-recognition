import os
import onnx
import torch
import numpy as np
import onnxruntime
import config as cfg
from torch import nn
from PIL import Image
from net.networks import CRNN
from _utils.generate import Generator
from _utils.utils import image_mark, remove_duplicate

class CRNN_ONNX:
    def __init__(self,
                 device,
                 batch_size: int,
                 input_shape: tuple,
                 opset_version: int,
                 torch_model_path: str,
                 onnx_model_path: str,
                 input_dynamic_axes: list=None,
                 output_dynamic_axes: list=None,
                 **kwargs):
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.torch_model_path = torch_model_path
        self.onnx_model_path = onnx_model_path
        self.input_dynamic_axes = input_dynamic_axes
        self.output_dynamic_axes = output_dynamic_axes

        self.ort_session = self.load_onnx()

    def load_torch_model(self):

        model = CRNN()
        if self.device:
            model = model.to(self.device)

        try:
            ckpt = torch.load(self.torch_model_path)
            model.load_state_dict(ckpt['state_dict'])
            print("model successfully loaded, ctc loss {:.3f}".format(ckpt['loss']))
        except:
            raise ("please enter the right params path")

        model = model.eval()

        return model

    def genearte_onnx(self):

        model = self.load_torch_model()

        input_names = ["input"]
        output_names = ["output"]
        input = torch.randn(self.input_shape).to(self.device)

        for named_param in model.named_parameters():
            name, param = named_param
            if param.requires_grad:
                input_names.append(name)

        torch.onnx.export(model, input, os.path.join(self.onnx_model_path, "crnn.onnx"),
                          verbose=True, input_names=input_names, output_names=output_names,
                          opset_version=self.opset_version, dynamic_axes=None)

    def load_onnx(self):

        try:
            ort_session = onnxruntime.InferenceSession(os.path.join(self.onnx_model_path, "crnn.onnx"),
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])
        except onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile:
            self.genearte_onnx()
            ort_session = onnxruntime.InferenceSession(os.path.join(self.onnx_model_path, "crnn.onnx"),
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])
        print("onnx model successfully loaded!")

        return ort_session

    def __call__(self, sources: np.ndarray):

        assert sources.shape[0] == self.batch_size

        outputs = self.ort_session.run(
            [_.name for _ in self.ort_session.get_outputs()],
            {self.ort_session.get_inputs()[0].name: sources.astype(np.float32)},
        )

        logits = outputs[0]

        return logits


if __name__ == '__main__':

    onnx_model = CRNN_ONNX(device=cfg.device,
                           batch_size=4,
                           input_shape=(4, 3, 32, 100),
                           opset_version=9,
                           torch_model_path=os.path.join(cfg.ckpt_path, '模型文件'),
                           onnx_model_path=os.path.join(cfg.onnx_path))

    gen = Generator(root_path=cfg.root_path,
                        training_path=cfg.training_path,
                        validate_path=cfg.validate_path,
                        cropped_size=cfg.cropped_size,
                        batch_size=4)

    validate_func = gen.generate(training=False)

    for i in range(gen.get_val_len()):

        sources, targets = next(validate_func)

        logits = onnx_model(sources)

        random_idx = np.random.choice(logits.shape[1], 1)

        source = sources[random_idx].squeeze().transpose(1, 2, 0)
        image = Image.fromarray(np.uint8((source + 1) * 127.5))

        logit = logits.transpose(1, 0, 2).argmax(axis=-1)[random_idx]
        logit = remove_duplicate(logit, cfg.target_size)

        image_mark(image, logit, 1)

