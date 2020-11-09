# -*- coding: utf-8 -*-
import argparse
import concurrent.futures
import math
import multiprocessing

import cv2
import os
import typing as typ
import pathlib as pth

import utilities.darknet as darknet

from utilities.log import logger_init


class VideoParser(object):
    def __init__(
            self,
            video: typ.Union[str, pth.Path, cv2.VideoCapture],
            batch_size: int
    ):
        # self.darknet_lib = darknet_lib_instance
        self.video = self.check_video_file_class(video)
        self.batch_size = batch_size
        self.network = self.net_init()
        self.meta = self.meta_init()
        self.darknet_image = self.darknet_image_conf()

    @staticmethod
    def check_path(v_pth) -> bool:
        v_path = pth.Path(v_pth)
        return os.path.exists(v_path)

    @staticmethod
    def capture_video_file(video_path: str) -> typ.Union[None, pth.Path]:
        # TODO not use delete
        if VideoParser.check_path(video_path):
            return pth.Path(video_path)
        else:
            log.error(f"Video file {video_path} not exist!")
            raise OSError(f"Video file {video_path} not exist!")

    @staticmethod
    def meta_init() -> darknet.load_meta:
        meta_path = os.environ.get("META_PATH")
        return darknet.load_meta(
            meta_path.encode("ascii")
        )

    def net_init(self) -> darknet.load_net_custom:
        # print(type(darknet))
        conf_path = os.environ.get("CONFIG_PATH")
        weight_path = os.environ.get("WEIGHT_PATH")
        return darknet.load_net_custom(
            conf_path.encode("ascii"),
            weight_path.encode("ascii"),
            0,
            self.batch_size
        )

    def darknet_image_conf(self) -> darknet.load_image:
        return darknet.make_image(
            darknet.network_width(self.network),
            darknet.network_height(self.network), 3)

    def frame_processing(self, frame) -> typ.List[typ.Tuple[str, str, typ.Any]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.network),
                                    darknet.network_height(self.network)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        return darknet.detect_image(
            self.network,
            self.meta.names.contents.value,
            self.darknet_image,
            thresh=float(os.environ.get("IMAGE_THRESHOLD", 0.25))
        )

    @staticmethod
    def check_video_file_class(video):
        if isinstance(video, cv2.VideoCapture):
            return video
        else:
            log.error(f"Video should be captured before processing! "
                      f"cap=cv2.VideoCapture(<path to video file>)")
            raise ValueError("Video should be captured before processing! "
                             "cap=cv2.VideoCapture(<path to video file>)")

    def video_frames_generator(self) -> typ.Generator:
        log.info(f"Frame reading is started!")
        while self.video.isOpened():
            frame_exist, frame = self.video.read()
            if frame_exist:
                yield frame
            else:
                log.info(f"Frame reading is finished!")
                break

        # if self.video.isOpened():
        #     print(dir(self.video.read))
        #     return (
        #         frame
        #         for frame_exist, frame
        #         in self.video.read()
        #         if frame_exist
        #     )
        # else:
        #     raise OSError("Video file not open!")

    def video_frames(self) -> typ.Generator:
        log.info(f"Start video frame enumerator!")
        c_video = self.video_frames_generator()
        return ((index, item) for index, item in enumerate(c_video, 1))
        # return (obj for obj in enumerate(c_video))
        # return (obj for obj in c_video)

    def processors_count(self) -> int:
        cpu = multiprocessing.cpu_count()
        available_cpu = cpu-2
        if available_cpu <= 0:
            return 1
        else:
            return available_cpu

    def multiprocess_video_parser(self):
        frames = self.video_frames()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            log.info("Image processing start!")
            future_to_args = {
                executor.submit(self.frame_processing, image_arg[1]):
                    image_arg for image_arg in frames
            }
            log.info("Image processing start!")
            log.info("Show results!")
            for future in concurrent.futures.as_completed(future_to_args):
                image_arg = future_to_args[future]
                try:
                    future.result()
                except Exception as exc:
                    log.error(f"frame process raised error!: {exc}")
                else:
                    for detection in image_arg:
                        x, y, w, h = detection[2][0], \
                                     detection[2][1], \
                                     detection[2][2], \
                                     detection[2][3]
                    log.info(f"Returned params --> X: {x}, Y: {y}, W: {w}, H: {h}")
            log.info("Program finish!")

    def frames_result(self):
        frames = self.video_frames()
        for i, j in frames:
            log.info(i)

    def mp_result(self, multi: bool = True):
        self.multiprocess_video_parser()

    def process_use_darknet_threads(self):
        frames = self.video_frames()
        for ind, frame in frames:
            # log.info(i)
            log.info(f"Start to processed frame # {ind}")
            res = self.frame_processing(frame)
            log.info(f"Finish to processed frame # {ind}")
            for detection in res:
                x, y, w, h = detection[2][0], \
                             detection[2][1], \
                             detection[2][2], \
                             detection[2][3]
                log.info(f"Returned params --> X: {x}, Y: {y}, W: {w}, H: {h}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis video.')
    parser.add_argument('-v', '--video', action="store", dest="video", required=True)
    parser.add_argument(
        '-b',
        '--batchsize',
        action="store",
        type=int,
        dest="batchsize",
        default=1,
    )
    parser.add_argument('-l', '--logfile', action="store", dest="logfile")
    arguments = parser.parse_args()
    if arguments.logfile:
        log = logger_init(file=pth.Path(arguments.logfile))
    else:
        log = logger_init()
    cap = cv2.VideoCapture(arguments.video)
    obj = VideoParser(cap, arguments.batchsize)
    # obj.frames_result()
    obj.process_use_darknet_threads()
    cap.release()

