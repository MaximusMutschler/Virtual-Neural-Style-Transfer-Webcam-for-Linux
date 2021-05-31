import gc
import os.path
import re

import cv2
import numpy as np
import onnx
# noinspection PyUnresolvedReferences
import pycuda.autoinit  # important for tensorrt to work
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.onnx
from torchvision import transforms

from style_transfer.transformer_net import TransformerNet

TRT_LOGGER = trt.Logger(min_severity=trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class StyleTransfer:
    def __init__(self, style_model_path="style_transfer/saved_models/style1.model", device="cuda",
                 cam_resolution=(720, 1280)):
        self.min_scale_factor = 0.1
        self.max_scale_factor = 1.6
        self.device = device
        self.style_model_weights_path = style_model_path
        self.default_input_shape = [1, 3, *cam_resolution]
        self.style_model = TransformerNet()
        self._create_tensorrt_network_and_config()
        self.trt_context = None
        self.trt_engine = None
        self.load_model(style_model_path)
        self._load_model_internal()
        self.is_new_model = False


    def load_model(self, style_model_path):
        self.is_new_model = True
        self.style_model_weights_path = style_model_path

    def optimize_model(self, modelpath):
        basepath = "".join(modelpath.split(".")[:-1])
        onnx_path = "." + basepath + ".onnx"
        trt_engine_path = "." + basepath + ".trtengine"
        trt_network = self.trt_builder.create_network(EXPLICIT_BATCH)
        if not (os.path.isfile(onnx_path) and os.path.isfile(trt_engine_path)):
            style_model = TransformerNet()
            self._load_weights_into_model(modelpath, style_model)
            self._optimize_model_internal(style_model, modelpath, onnx_path, trt_engine_path, trt_network)
            torch.cuda.empty_cache()

    def _optimize_model_internal(self, style_model, modelpath, onnx_path, trt_engine_path, trt_network):
        print("optimizing", modelpath)
        self._save_model_to_onnx(style_model, path=onnx_path)
        parser = trt.OnnxParser(trt_network, TRT_LOGGER)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    if os.getuid() == 0:
                        os.chmod(onnx_path, 0o0777)
        engine = self.trt_builder.build_engine(trt_network, self.trt_config)
        if engine is None:
            raise Exception("engine is none")

        context = engine.create_execution_context()
        print("saving tensorrt engine to ", trt_engine_path)
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())
            if os.getuid() == 0:
                os.chmod(trt_engine_path, 0o0777)
        return engine, context

    @staticmethod
    def _load_weights_into_model(style_model_weights_path, style_model):
        state_dict = torch.load(style_model_weights_path)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)

    def _load_model_internal(self):
        # this only works if called form the main thread!
        del self.trt_engine
        del self.trt_context
        gc.collect()
        self._load_weights_into_model(self.style_model_weights_path, self.style_model)
        # optimize
        base_path = "".join(self.style_model_weights_path.split(".")[:-1])

        onnx_path = "." + base_path + ".onnx"
        trt_engine_path = "." + base_path + ".trtengine"
        trt_network = self.trt_builder.create_network(EXPLICIT_BATCH)
        if not (os.path.isfile(onnx_path) and os.path.isfile(trt_engine_path)):
            print("optimizing model for your graphics card. This might take some time.")
            engine, context = self._optimize_model_internal(self.style_model, self.style_model_weights_path, onnx_path,
                                                            trt_engine_path, trt_network)
        else:
            # this has to be done otherwise deserialize_cuda_engine does not work
            parser = trt.OnnxParser(trt_network, TRT_LOGGER)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

        self.trt_engine = engine
        self.trt_context = context
        self.is_new_model = False

    def __del__(self):
        del self.trt_engine
        del self.trt_context

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def _save_model_to_onnx(self, model, example_input=None, path="./test.onnx"):

        if example_input is None:
            example_input = torch.ones(
                *self.default_input_shape)
        torch.onnx.export(
            model,
            example_input,
            path,
            export_params=True,
            # do_constant_folding=True,
            input_names=['input'],  # Pass names as per model input name
            output_names=['output'],  ## Pass names as per model output name
            opset_version=10,  # export the model to the  opset version of the onnx submodule.
            dynamic_axes={  # this will makes export more generalize to take batch for prediction
                'input': [2, 3],
                # 'output': {0: 'batch'},
            }

        )
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print("saved model onnx to: ", path)

    @staticmethod
    def _resize_crop(image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
        h, w = (h // 8) * 8, (w // 8) * 8
        image = image[:h, :w, :]
        return image

    def _create_tensorrt_network_and_config(self):

        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        shape = np.array(self.default_input_shape)
        dynamic_dim = shape[2:4]
        fix_dim = shape[0:2]
        min_shape = np.array([*fix_dim, *(dynamic_dim * self.min_scale_factor)]).astype(int)
        max_shape = np.array([*fix_dim, *(dynamic_dim * self.max_scale_factor)]).astype(int)
        optimization_shape = np.array(self.default_input_shape).astype(int)
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
        profile.set_shape("input", min_shape, optimization_shape, max_shape)
        profile.set_shape("output", min_shape, optimization_shape, max_shape)
        config.add_optimization_profile(profile)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # force FP16
        self.trt_config = config
        self.trt_builder = builder

    def stylize(self, frame):
        if self.is_new_model:
            self._load_model_internal()

        content_image = self._resize_crop(frame)
        content_image = content_image.astype(np.float32)  # / 127.5 - 1
        content_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        content_image = content_transform(content_image).unsqueeze(0)

        engine = self.trt_engine
        context = self.trt_context
        stream = cuda.Stream()
        context.set_optimization_profile_async(0, stream.handle)

        context.set_binding_shape(0, content_image.shape)
        input_shape = context.get_binding_shape(0)
        input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
        device_input = cuda.mem_alloc(input_size)

        output_shape = context.get_binding_shape(1)
        host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
        device_output = cuda.mem_alloc(host_output.nbytes)

        host_input = np.array(content_image, dtype=np.float32, order='C')

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(device_input, host_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
        # https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
        output_data = np.array(host_output).reshape(engine.max_batch_size, *output_shape[1:])

        output = np.squeeze(output_data)
        output = np.moveaxis(output, 0, 2)
        red = output[:, :, 2].copy()
        green = output[:, :, 1].copy()
        blue = output[:, :, 0].copy()
        output[:, :, 0] = red
        output[:, :, 1] = green
        output[:, :, 2] = blue

        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
