import re
import onnx
import onnxruntime
import cv2
import numpy as np
import torch
import torch.onnx
from torchvision import transforms

from style_transfer.transformer_net import TransformerNet


class StyleTransfer():
    def __init__(self, style_model_path="style_transfer/saved_models/style1.model", device="cuda"):

        self.device = device
        self.style_model_path = style_model_path
        self.style_model = TransformerNet()
        self.state_dict = torch.load(self.style_model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(self.state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del self.state_dict[k]
        self.style_model.load_state_dict(self.state_dict)
        self.style_model.to(device)
        #TODO if not already there
        onnx_model_path="./test.onnx"
        self._save_model_to_onnx(self.style_model,path=onnx_model_path)
        self.onnx_session= onnxruntime.InferenceSession(onnx_model_path)


    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    @staticmethod
    def _save_model_to_onnx(model,example_input=None,path="./test.onnx"):

        if example_input==None:
            example_input=torch.ones(1,3,720,1280,device='cuda') # TODO bind memory to gpu for onnx : https://github.com/microsoft/onnxruntime/issues/2750
        torch.onnx.export(
            model,  ## pass model
            (example_input),  ## pass inpout example
            path,  ##output path
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],  ## Pass names as per model input name
            output_names=['output'],  ## Pass names as per model output name
            opset_version=11,  ##  export the model to the  opset version of the onnx submodule.
            # dynamic_axes={  ## this will makes export more generalize to take batch for prediction
            #     'input': [ 2, 3],
            #     'output': {0: 'batch'},
            # }

        )
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)



    @staticmethod
    def resize_crop(image):
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

    def stylize(self, frame):
        content_image = self.resize_crop(frame)
        content_image = content_image.astype(np.float32)  # / 127.5 - 1
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image).unsqueeze(0)


        #content_image2 = content_image2.unsqueeze(0).to(self.device)

        #with torch.no_grad():
        #    output = self.style_model(content_image).cpu().numpy()

        onnx_inputs = {self.onnx_session.get_inputs()[0].name: (content_image)}
        oonnx_outs = self.onnx_session.run(None, onnx_inputs)


        output = np.squeeze(oonnx_outs)
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
