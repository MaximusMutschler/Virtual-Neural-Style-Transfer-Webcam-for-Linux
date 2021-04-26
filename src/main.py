import signal
import sys
from argparse import ArgumentParser
from functools import partial

import pynput.keyboard as keyboard

from facecam import FakeCam


def parse_args():
    parser = ArgumentParser(description="Applying stylees to your web cam image under \
                            GNU/Linux. For more information, please refer to: \
                            TODO")
    parser.add_argument("-W", "--width", default=1280, type=int,
                        help="Set real webcam width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Set real webcam height")
    parser.add_argument("-F", "--fps", default=30, type=int,
                        help="Set real webcam FPS")
    parser.add_argument("-C", "--codec", default='MJPG', type=str,
                        help="Set real webcam codec")
    parser.add_argument("-S", "--scale-factor", default=1.0, type=float,
                        help="Scale factor of the image sent the neural network")  # TODO make this dynamical *1.1
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-v", "--akvcam-path", default="/dev/video3",
                        help="virtual akvcam output device path")
    parser.add_argument("--cartoonize", action="store_true",
                        help="use cartoon style transfer from https://github.com/SystemErrorWang/White-box-Cartoonization")
    parser.add_argument("-s", "--style-model-dir", default="./data/style_transfer_models",
                        help="Folder which (subfolders) contains saved style transfer networks. Have to end with '.model' or '.pth'. Own styles created with https://github.com/pytorch/examples/tree/master/fast_neural_style can be used.")
    parser.add_argument("-c", "--cartoonize-model-dir", default="./data/cartoonize_models",
                        help="Folder which (subfolders) contains saved cartoonize networks. A .index file has to exist for each checkpoint.  Own styles created with https://github.com/SystemErrorWang/White-box-Cartoonization can be used.")

    return parser.parse_args()



def main():
    args = parse_args()
    cam = FakeCam(
        fps=args.fps,
        width=args.width,
        height=args.height,
        codec=args.codec,
        scale_factor=args.scale_factor,
        webcam_path=args.webcam_path,
        akvcam_path=args.akvcam_path,
        is_cartoonize=args.cartoonize,
        cartoonize_model_dir=args.cartoonize_model_dir,
        style_model_dir=args.style_model_dir,
    )

    def sig_interrupt_handler(signal, frame, cam):
        print("Stopping fake cam process")
        cam.stop()


    signal.signal(signal.SIGINT, partial(sig_interrupt_handler,cam=cam))

    keyboard.GlobalHotKeys({
        '<ctrl>+1': cam.switch_is_styling,
        '<ctrl>+2': cam.set_previous_style,
        '<ctrl>+3': cam.set_next_style,
        '<ctrl>+4': partial(cam.add_to_scale_factor,-0.1),
        '<ctrl>+5': partial(cam.add_to_scale_factor,0.1),
    }).start()

    # keyboard.KeyboardListener(on_press=on_press).start()
    print("Running...")
    print("Press CTRL-1 to deactivate and activate styling")
    print("Press CTRL-2 to load the previous style")
    print("Press CTRL-3 to load the next style")
    print("Press CTRL-4 to decrease the scale factor of the model input")
    print("Press CTRL-5 to increase the scale factor of the model input")
    print("Please CTRL-c to exit")

    cam.run()  # loops

    print("exit 0")

    sys.exit(0)



if __name__ == "__main__":
    main()
