import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import signal
import sys
from argparse import ArgumentParser
from functools import partial
import threading
from fakecam import FakeCam

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
    parser.add_argument("-S", "--scale-factor", default=0.7, type=float,
                        help="Scale factor of the image sent the neural network")
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-v", "--akvcam-path", default="/dev/video13",
                        help="virtual akvcam output device path")
    parser.add_argument("-s", "--style-model-dir", default="./data/style_transfer_models_bu",
                        help="Folder which (subfolders) contains saved style transfer networks. Have to end with '.model' or '.pth'. Own styles created with https://github.com/pytorch/examples/tree/master/fast_neural_style can be used.")
    parser.add_argument("-n", "--noise-suppressing", default=25.0, type=float,
                        help="higher values reduce noise introduced by the style transfer but might lead to skewed human faces")
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
        style_model_dir=args.style_model_dir,
        noise_suppressing_factor=args.noise_suppressing,
    )

    print("Running...")
    print("Enter 1+BACKSPACE to deactivate and activate styling")
    print("Enter 2+BACKSPACE to load the previous style")
    print("Enter 3+BACKSPACE to load the next style")
    print("Enter 4+BACKSPACE to decrease the scale factor of the model input")
    print("Enter 5+BACKSPACE to increase the scale factor of the model input")
    print("Please CTRL-c to exit")

    def listen_for_input():
        t = threading.currentThread()
        while True:
            input_ = input()
            if input_ == "1":
                cam.switch_is_styling()
            elif input_ == "2":
                cam.set_previous_style()
            elif input_ == "3":
                cam.set_next_style()
            elif input_ == "4":
                cam.add_to_scale_factor(-0.1)
            elif input_ == "5":
                cam.add_to_scale_factor(0.1)
            elif input_ == "6":
                cam.add_to_noise_factor(-5)
            elif input_ == "7":
                cam.add_to_noise_factor(5)
            else:
                print("input {} was not recognized".format(input_))
            time.sleep(1)

    listen_thread = threading.Thread(target=listen_for_input, daemon=True)
    listen_thread.start()

    def sig_interrupt_handler(signal_, frame_, cam_):
        print("Stopping fake cam process")
        cam_.stop()

    signal.signal(signal.SIGINT, partial(sig_interrupt_handler, cam_=cam))

    cam.run()  # loops

    print("exit 0")

    sys.exit(0)


if __name__ == "__main__":
    main()

# TODO make deepfake version https://www.youtube.com/watch?v=mUfJOQKdtAk
