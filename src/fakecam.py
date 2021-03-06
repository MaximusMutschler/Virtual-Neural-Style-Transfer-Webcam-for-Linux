import os
import threading
import time

import cv2
import numpy as np

from akvcam import AkvCameraWriter
from realcam import RealCam
from style_transfer.neural_style import StyleTransfer


class FakeCam:
    def __init__(
            self,
            fps: int,
            width: int,
            height: int,
            codec: str,
            scale_factor: float,
            webcam_path: str,
            akvcam_path: str,
            style_model_dir: str,
            noise_suppressing_factor: float,
    ) -> None:
        self.check_webcam_existing(webcam_path)
        self.check_webcam_existing(akvcam_path)
        self.scale_factor = scale_factor
        self.real_cam = RealCam(webcam_path, width, height, fps, codec)
        # In case the real webcam does not support the requested mode.
        self.width = self.real_cam.get_frame_width()
        self.height = self.real_cam.get_frame_height()
        self.fake_cam_writer = AkvCameraWriter(akvcam_path, self.width, self.height)
        self.style_number = 0
        self.model_dir = style_model_dir
        self.styler_lock = threading.Lock()
        self.is_stop = False
        self.styler = None
        self.set_style_number(self.style_number)
        self.is_styling = True
        self.optimize_models()
        self.current_fps = 0
        self.last_frame = None
        self.noise_epsilon = noise_suppressing_factor

    @staticmethod
    def check_webcam_existing(path):
        if not os.path.exists(path):
            error = "cam path not existing: " + path
            print(error)
            raise Exception(error)

    def put_frame(self, frame):
        self.fake_cam_writer.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        with self.styler_lock:
            self.is_stop = True

    def run(self):
        self.real_cam.start()
        t0 = time.monotonic()
        print_fps_period = 5.0
        frame_count = 0
        while not self.is_stop:
            current_frame = self.real_cam.read()
            if current_frame is None:
                # print("frame none")
                time.sleep(0.1)
                continue

            with self.styler_lock:
                current_frame = cv2.resize(current_frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
                if self.is_styling:
                    current_frame = self._supress_noise(current_frame)
                    try:
                        current_frame = self.styler.stylize(current_frame)
                    except Exception as e:
                        print("error during style transfer", e)
                        pass
            self.put_frame(current_frame)
            frame_count += 1
            td = time.monotonic() - t0
            #print(td)
            if td > print_fps_period:
                self.current_fps = frame_count / td
                print("\r (FPS: {:6.2f}) Waiting for input: ".format(self.current_fps), end=" ")
                frame_count = 0
                t0 = time.monotonic()
        print("stopped fake cam")
        self.real_cam.stop()
        self.fake_cam_writer.stop()

    def _supress_noise(self, current_frame):
        if self.last_frame is not None and self.last_frame.shape == current_frame.shape:
            delta = np.abs(self.last_frame - current_frame) <= self.noise_epsilon
            current_frame[delta] = self.last_frame[delta]
        self.last_frame = current_frame
        return current_frame

    def _get_list_of_all_models(self, model_dir, file_endings=[".index", ".pth", ".model"]):
        list_of_paths = []
        for dir_path, dir_name, file_names in os.walk(model_dir):
            for file_name in file_names:
                for file_ending in file_endings:
                    if file_name.endswith(file_ending):
                        list_of_paths.append(os.path.join(dir_path, file_name))
                        break
                if len(file_endings) == 0:
                    list_of_paths.append(os.path.join(dir_path, file_name))
        list_of_paths.sort()
        return list_of_paths

    def add_to_scale_factor(self, addend=0.1):
        proposed_scale_factor = round(self.scale_factor + addend, 1)
        if proposed_scale_factor <= 0:
            print("scale factor cannot be smaller than 0")
        # elif self.scale_factor+addend > 2.0:
        #     print("a scale factor larger than 2.0")
        else:
            with self.styler_lock:
                self.scale_factor = proposed_scale_factor
                print("new scale factor is: ", self.scale_factor)

    def add_to_noise_factor(self, addend=5):
        proposed_noise_factor = round(self.noise_epsilon + addend, 1)
        if proposed_noise_factor <= 0:
            print("noise factor cannot be smaller than 0")
        else:
            with self.styler_lock:
                self.noise_epsilon = proposed_noise_factor
                print("new noise factor is: ", self.noise_epsilon)

    def set_next_style(self):
        model_paths = self._get_list_of_all_models(self.model_dir)
        number = self.style_number
        if self.style_number + 1 > len(model_paths) - 1:
            number = 0
        else:
            number += 1
        self.set_style_number(number, model_paths)

    def set_previous_style(self):
        model_paths = self._get_list_of_all_models(self.model_dir)
        number = self.style_number
        if self.style_number - 1 == -1:
            number = len(model_paths) - 1
        else:
            number -= 1
        self.set_style_number(number, model_paths)

    def optimize_models(self):
        print("-" * 50)
        print("optimizing models for your graphics card. This might take several minutes for the first time.")
        print("-" * 50)
        model_paths = self._get_list_of_all_models(self.model_dir)
        for model_path in model_paths:
            self.styler.optimize_model(model_path)

    def set_style_number(self, number, model_paths=None):
        if model_paths is None:
            model_paths = self._get_list_of_all_models(self.model_dir)
        # if self.style_number == number:
        #     print("style already set")
        #     return
        if number < len(model_paths) and number > -1:
            model_path = self._get_list_of_all_models(self.model_dir)[number]
            try:
                with self.styler_lock:
                    if self.styler is None:
                        self.styler = StyleTransfer(model_path)
                    else:
                        self.styler.load_model(model_path)
                self.style_number = number
                print("model changed to:", model_path)
            except Exception as e:
                # print("style model could not be changed".format(number), e)
                raise e
        else:
            print("model with number {} does not exist".format(number))

    def switch_is_styling(self):
        with self.styler_lock:
            if not self.is_styling:
                self.is_styling = True
                print("styling activated")
            else:
                self.is_styling = False
                print("styling deactivated")

    # speed test style transfer:
    # gpu pytorch  11.6
    # gpu onnx     ca 3.0 FAIL!!!
    # gpu tensorrt 16.8 with float16: 22
    # cpu pytorch  0.45 fps
    # cpu onnx     0.75 fps
