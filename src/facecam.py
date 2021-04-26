import os
import threading
import time

import cv2

from akvcam import AkvCameraWriter
from realcam import RealCam


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
            is_cartoonize: bool,
            cartoonize_model_dir: str,
            style_model_dir: str,
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
        self.is_cartoonize = is_cartoonize
        if self.is_cartoonize:
            self.model_dir = cartoonize_model_dir
        else:
            self.model_dir = style_model_dir
        self.styler_lock = threading.Lock()
        self.scale_factor_lock = threading.Lock()
        self.stop_lock = threading.Lock()
        self.is_stop = False
        self.styler = None
        self.set_style_number(self.style_number)

    def check_webcam_existing(self, path):
        if not os.path.exists(path):
            error = "cam path not existing: " + path
            print(error)
            raise Exception(error)

    def put_frame(self, frame):
        self.fake_cam_writer.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        with self.stop_lock:
            self.is_stop = True




    def run(self):
        self.real_cam.start()
        t0 = time.monotonic()
        print_fps_period = 1
        frame_count = 0
        while not self.is_stop:
            current_frame = self.real_cam.read()
            if current_frame is None:
                time.sleep(0.1)
                continue
            current_frame = cv2.resize(current_frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            if self.styler is not None:
                try:
                    # frame_buffer=frame_buffer
                    # current_frame = self.cartoonizer.cartoonize_frame([current_frame])[0] # todo higher batch size does not help
                    current_frame = self.styler.stylize(current_frame)
                except:
                    pass
            self.put_frame(current_frame)
            frame_count += 1
            td = time.monotonic() - t0
            if td > print_fps_period:
                self.current_fps = frame_count / td
                print("FPS: {:6.2f}".format(self.current_fps), end="\r")
                frame_count = 0
                t0 = time.monotonic()
        print("stopped fake cam")
        self.real_cam.stop()
        self.fake_cam_writer.stop()

    def _get_list_of_all_models(self, model_dir, file_endings=[".index", ".pth", ".model"]):
        list_of_paths = []
        for dir_path, dir_name, file_names in os.walk(model_dir):
            for file_name in file_names:
                for file_ending in file_endings:
                    if (file_name.endswith(file_ending)):
                        list_of_paths.append(os.path.join(dir_path, file_name))
                        break
                if len(file_endings) == 0:
                    list_of_paths.append(os.path.join(dir_path, file_name))
        list_of_paths.sort()
        return list_of_paths

    #
    # def scale_scale_factor(self,factor=1.1):
    #     with self.scale_factor_lock:
    #         self.scale_factor*=factor

    def add_to_scale_factor(self, addend=0.1):
        proposed_scale_factor = round(self.scale_factor + addend, 1)
        if proposed_scale_factor <= 0:
            print("scale factor cannot be smaller than 0")
        # elif self.scale_factor+addend > 2.0:
        #     print("a scale factor larger than 2.0")
        else:
            with self.scale_factor_lock:
                self.scale_factor = proposed_scale_factor
                print("new scale factor is: ", self.scale_factor)

    def set_next_style(self):
        model_paths = self._get_list_of_all_models(self.model_dir)
        number = self.style_number
        if self.style_number + 1 > len(model_paths) - 1:
            number= 0
        else:
            number += 1
        self.set_style_number(number, model_paths)

    def set_previous_style(self):
        model_paths = self._get_list_of_all_models(self.model_dir)
        number=self.style_number
        if self.style_number - 1 == -1:
            number = len(model_paths) - 1
        else:
            number -= 1
        self.set_style_number(number, model_paths)

    def set_style_number(self, number, model_paths=None):
        if model_paths == None:
            model_paths = self._get_list_of_all_models(self.model_dir)
        # if self.style_number == number:
        #     print("style already set")
        #     return
        if number < len(model_paths) and number > -1:
            model_path = self._get_list_of_all_models(self.model_dir)[number]
            try:
                if self.is_cartoonize:
                    from white_box_cartoonization.cartoonize import Cartoonizer
                    with self.styler_lock:
                        self.styler = Cartoonizer(model_path=model_path)
                else:
                    from style_transfer.neural_style import StyleTransfer
                    with self.styler_lock:
                        self.styler = StyleTransfer(style_model_path=model_path)
                self.style_number = number
                print("model changed to:", model_path)
            except Exception as e:
                print("style model could not be changed".format(number), e)
        else:
            print("model with number {} does not exist".format(number))

    def switch_is_styling(self):
        if self.styler is None:
            self.set_style_number(self.style_number)
        else:
            self.styler=None
            print("disabled styling")


