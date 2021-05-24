import os.path
import re
import onnx
import onnxruntime
import cv2
import torch
import torch.onnx
from torchvision import transforms
import pycuda.driver as cuda
import pycuda.autoinit #important for tensorrt to work
import numpy as np
import tensorrt as trt
import gc

from style_transfer.transformer_net import TransformerNet

#TRT_LOGGER = trt.Logger(min_severity=trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger(min_severity=trt.Logger.WARNING)

class StyleTransfer():
    def __init__(self, style_model_path="style_transfer/saved_models/style1.model", device="cuda", cam_resolution=(720,1280)):
        self.min_scale_factor=0.1
        self.max_scale_factor=1.6
        self.device = device
        self.style_model_weights_path = style_model_path
        self.default_input_shape=[1,3,*cam_resolution]
        self.style_model = TransformerNet()
        #self.style_model.to(device)

        self._create_tensorrt_network_and_config()
        self.trt_context=None
        self.trt_engine=None
        self.load_model(style_model_path)
        self._load_model_internal()
        self.is_new_model = False



        #self.onnx_session= onnxruntime.InferenceSession(onnx_model_path)

        #self.tensorrt_engine, self.tesorrt_context = self._build_engine(onnx_model_path)
    def load_model(self,style_model_path):
        self.is_new_model=True
        self.style_model_weights_path=style_model_path

    def optimize_model(self, modelpath):
        basepath= "".join(modelpath.split(".")[:-1])
        onnx_path = "."+ basepath + ".onnx"
        trt_engine_path ="."+ basepath + ".trtengine"
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt_network = self.trt_builder.create_network(EXPLICIT_BATCH)
        if not (os.path.isfile(onnx_path) and os.path.isfile(trt_engine_path)):
            style_model=TransformerNet()
            self._load_weights_into_model(modelpath, style_model)
            self._optimize_model_internal(style_model,modelpath,onnx_path,trt_engine_path,trt_network)


    def _optimize_model_internal(self,style_model,modelpath,onnx_path,trt_engine_path,trt_network):
        print("optimizing", modelpath)
        self._save_model_to_onnx(style_model, path=onnx_path)
        parser = trt.OnnxParser(trt_network, TRT_LOGGER)
        with open(onnx_path, 'rb') as model:
            #print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        #print('Completed parsing of ONNX file')
        #print('Building an tensorrt engine. This might take some time...')
        engine = self.trt_builder.build_engine(trt_network, self.trt_config)
        context = engine.create_execution_context()
        #print("Completed creating Engine")
        print("saving tensorrt engine to ", trt_engine_path)
        with open(trt_engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine,context

    def _load_weights_into_model(self, style_model_weights_path, style_model):
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
        # load model weights
        self._load_weights_into_model(self.style_model_weights_path,self.style_model)

        # optimize
        basepath= "".join(self.style_model_weights_path.split(".")[:-1])

        onnx_path = "."+ basepath + ".onnx"
        trt_engine_path ="."+ basepath + ".trtengine"
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt_network = self.trt_builder.create_network(EXPLICIT_BATCH)
        if not (os.path.isfile(onnx_path) and os.path.isfile(trt_engine_path)):
            print("optimizing model to your graphics card, this might take some time.")
            engine,context=self._optimize_model_internal(self.style_model,self.style_model_weights_path,onnx_path,trt_engine_path,trt_network)

            #with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            #        engine = runtime.deserialize_cuda_engine(f.read())

#            runtime = trt.Runtime(TRT_LOGGER)
            # serialized_engine=engine.serialize()
            # engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            # this has to be done otherwise deserialize_cuda_engine does not work
            parser = trt.OnnxParser(trt_network, TRT_LOGGER)
            with open(onnx_path, 'rb') as model:
                #print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()


        self.trt_engine = engine
        self.trt_context = context
        self.is_new_model=False





    def __del__(self):
        del self.trt_engine
        del self.trt_context


    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def _save_model_to_onnx(self,model,example_input=None,path="./test.onnx"):

        if example_input==None:
            example_input=torch.ones(*self.default_input_shape) # TODO bind memory to gpu for onnx : https://github.com/microsoft/onnxruntime/issues/2750
        torch.onnx.export(
            model,  ## pass model
            (example_input),  ## pass inpout example
            path,  ##output path
            export_params=True,
            #do_constant_folding=True,
            input_names=['input'],  ## Pass names as per model input name
            output_names=['output'],  ## Pass names as per model output name
            opset_version=10,  ##  export the model to the  opset version of the onnx submodule.
            dynamic_axes={  ## this will makes export more generalize to take batch for prediction
                'input': [ 2, 3],
                #'output': {0: 'batch'},
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
        shape= np.array(self.default_input_shape)
        dynamic_dim=shape[2:4]
        fix_dim=shape[0:2]
        min_shape = np.array([*fix_dim,*(dynamic_dim*self.min_scale_factor)]).astype(int)
        max_shape = np.array([*fix_dim,*(dynamic_dim*self.max_scale_factor)]).astype(int)
        optimization_shape = np.array(self.default_input_shape).astype(int)
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
        profile.set_shape("input", min_shape,optimization_shape,max_shape)
        profile.set_shape("output", min_shape,optimization_shape,max_shape)
        config.add_optimization_profile(profile)

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)# force FP16
        self.trt_config = config
        self.trt_builder= builder


    def stylize(self, frame):
        if self.is_new_model:
            self._load_model_internal()


        content_image = self._resize_crop(frame)
        content_image = content_image.astype(np.float32)  # / 127.5 - 1
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image).unsqueeze(0)


        engine= self.trt_engine
        context= self.trt_context
        stream = cuda.Stream()
        context.set_optimization_profile_async(0, stream.handle)
        for i ,binding in enumerate(engine):
            if engine.binding_is_input(i):  # we expect only one input
                context.set_binding_shape(i,content_image.shape)
                input_shape_engine = engine.get_binding_shape(binding) # This has dynamic shapes
                input_shape = context.get_binding_shape(i)# this has correct shapes
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                device_input = cuda.mem_alloc(input_size)
            else:  # and one output
                #self.tesorrt_context.set_binding_shape(i, content_image.shape)
                output_shape_engine = engine.get_binding_shape(binding)
                output_shape = context.get_binding_shape(i)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
                device_output = cuda.mem_alloc(host_output.nbytes)



        host_input=np.array(content_image, dtype=np.float32, order='C')

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(device_input, host_input, stream)
        # Run inference.
        #engine.get_binding_index()


        context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        #TODO https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
        # https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
        output_data = np.array(host_output).reshape(engine.max_batch_size, *output_shape[1:])

        #ctx.pop()
        #del ctx

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

# def stylize_onnx_caffe2(content_image, args):
#     """
#     Read ONNX model and run it using Caffe2
#     """
#
#     assert not args.export_onnx
#
#     import onnx
#     import onnx_caffe2.backend
#
#     model = onnx.load(args.model)
#
#     prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
#     inp = {model.graph.input[0].name: content_image.numpy()}
#     c2_out = prepared_backend.run(inp)[0]
#
#     return torch.from_numpy(c2_out)

# 
# def main():
#     main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
#     subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
# 
#     train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
#     train_arg_parser.add_argument("--epochs", type=int, default=2,
#                                   help="number of training epochs, default is 2")
#     train_arg_parser.add_argument("--batch-size", type=int, default=4,
#                                   help="batch size for training, default is 4")
#     train_arg_parser.add_argument("--dataset", type=str, required=True,
#                                   help="path to training dataset, the path should point to a folder "
#                                        "containing another folder with all the training images")
#     train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
#                                   help="path to style-image")
#     train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
#                                   help="path to folder where trained model will be saved.")
#     train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
#                                   help="path to folder where checkpoints of trained models will be saved")
#     train_arg_parser.add_argument("--image-size", type=int, default=256,
#                                   help="size of training images, default is 256 X 256")
#     train_arg_parser.add_argument("--style-size", type=int, default=None,
#                                   help="size of style-image, default is the original size of style image")
#     train_arg_parser.add_argument("--cuda", type=int, required=True,
#                                   help="set it to 1 for running on GPU, 0 for CPU")
#     train_arg_parser.add_argument("--seed", type=int, default=42,
#                                   help="random seed for training")
#     train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
#                                   help="weight for content-loss, default is 1e5")
#     train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
#                                   help="weight for style-loss, default is 1e10")
#     train_arg_parser.add_argument("--lr", type=float, default=1e-3,
#                                   help="learning rate, default is 1e-3")
#     train_arg_parser.add_argument("--log-interval", type=int, default=500,
#                                   help="number of images after which the training loss is logged, default is 500")
#     train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
#                                   help="number of batches after which a checkpoint of the trained model will be created")
# 
#     eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
#     eval_arg_parser.add_argument("--content-image", type=str, required=True,
#                                  help="path to content image you want to stylize")
#     eval_arg_parser.add_argument("--content-scale", type=float, default=None,
#                                  help="factor for scaling down the content image")
#     eval_arg_parser.add_argument("--output-image", type=str, required=True,
#                                  help="path for saving the output image")
#     eval_arg_parser.add_argument("--model", type=str, required=True,
#                                  help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
#     eval_arg_parser.add_argument("--cuda", type=int, required=True,
#                                  help="set it to 1 for running on GPU, 0 for CPU")
#     eval_arg_parser.add_argument("--export_onnx", type=str,
#                                  help="export ONNX model to a given file")
# 
#     args = main_arg_parser.parse_args()
# 
#     if args.subcommand is None:
#         print("ERROR: specify either train or eval")
#         sys.exit(1)
#     if args.cuda and not torch.cuda.is_available():
#         print("ERROR: cuda is not available, try running on CPU")
#         sys.exit(1)
# 
#     if args.subcommand == "train":
#         check_paths(args)
#         train(args)
#     else:
#         stylize(args)
# 
# 
# if __name__ == "__main__":
#     main()
