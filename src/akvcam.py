import os
import threading
from fcntl import ioctl
from queue import Queue

import cv2

import v4l2


class AkvCameraWriter:
    def __init__(self, webcam, width, height):
        self.webcam = webcam
        self.width = width
        self.height = height
        self.d = self.open_camera()
        self.queue = Queue(maxsize=1)
        self.is_stop_lock = threading.Lock()
        self.is_stop = False
        self.thread = threading.Thread(target=self.writer_thread)
        self.thread.start()

    def open_camera(self):
        d = os.open(self.webcam, os.O_RDWR)
        cap = v4l2.v4l2_capability()
        ioctl(d, v4l2.VIDIOC_QUERYCAP, cap)
        vid_format = v4l2.v4l2_format()
        vid_format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        vid_format.fmt.pix.width = self.width
        vid_format.fmt.pix.height = self.height
        vid_format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_RGB24
        vid_format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        vid_format.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_SRGB
        ioctl(d, v4l2.VIDIOC_S_FMT, vid_format)
        return d

    def writer_thread(self):
        while not self.is_stop:
            try:
                elem = self.queue.get(timeout=1)
            except:
                # print("akvcam waited longer as 1 second for a frame. Continuing.")
                continue
            if elem is None:
                error = "input queue for akvcam was empty"
                raise Exception(error)
            image_data = cv2.resize(elem, (self.width, self.height)).tobytes()
            try:
                os.write(self.d, image_data)
            except Exception:
                error = "could not write image to akvcam output device"
                raise IOError(error)

    def stop(self):
        with self.is_stop_lock:
            self.is_stop = True
        if self.thread.is_alive():
            self.thread.join()
        os.close(self.d)
        print("stopped fake cam writer")

    def schedule_frame(self, image_):
        self.queue.put(image_)

    def __del__(self):
        os.close(self.d)


if __name__ == "__main__":
    camera_w, camera_h = 1280, 720  # must be defined as possible resolution in /etc/akvcam/config.ini
    writer = AkvCameraWriter("/dev/video3", camera_w, camera_h)
    image = cv2.imread("background.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    while True:
        writer.schedule_frame(image)
