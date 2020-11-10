# -*- coding: utf-8 -*-
import argparse
import pathlib as pth
import typing as typ
import os

import cv2
import numpy as np

import settings
from utilities import darknet

BATCH_SIZE = settings.BATCH_SIZE


class VideoParser(object):

    def __init__(
            self,
            batch_size: int,
            video_path: str,
            thresh: float,
            network: typ.Any,
            class_names: typ.Any,
            class_colors: typ.Any,
    ):
        self.batch_size = batch_size if \
            batch_size <= BATCH_SIZE else \
            BATCH_SIZE
        self.video_path = self.check_path(video_path)
        self.thresh = thresh
        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors

    @staticmethod
    def check_path(video_path: str) -> typ.Union[str, pth.Path, OSError]:
        if os.path.isfile(video_path):
            return video_path
        raise OSError(f"Path to video file incorrect or not exist. "
                      f"Path {video_path}")

    @staticmethod
    def check_batch_shape(images, batch_size):
        """
            Image sizes should be the same width and height
        """
        shapes = [image.shape for image in images if len(image)]
        if len(set(shapes)) > 1:
            raise ValueError("Images don't have same shape")
        if len(shapes) > batch_size:
            raise ValueError("Batch size higher than number of images")
        return shapes[0]

    @staticmethod
    def prepare_batch(images, network, channels=3) -> darknet.IMAGE:
        # print(f"Len Images: {len(images)}")
        width = darknet.network_width(network)
        height = darknet.network_height(network)

        darknet_images = []
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            custom_image = image_resized.transpose(2, 0, 1)
            darknet_images.append(custom_image)

        batch_array = np.concatenate(darknet_images, axis=0)
        batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
        darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
        return darknet.IMAGE(width, height, channels, darknet_images)

    def batch_detection(self, network, images, class_names, class_colors,
                        hier_thresh=.5, nms=.45):
        image_height, image_width, _ = self.check_batch_shape(images, self.batch_size)
        image_width = darknet.network_width(network)
        image_height = darknet.network_height(network)
        darknet_images = self.prepare_batch(images, network)
        batch_detections = darknet.network_predict_batch(
            network,
            darknet_images,
            self.batch_size,
            image_width,
            image_height,
            self.thresh,
            hier_thresh,
            None,
            0,
            0
        )
        batch_predictions = []
        for idx in range(self.batch_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                darknet.do_nms_obj(detections, num, len(class_names), nms)
            predictions = darknet.remove_negatives(detections, class_names, num)
            images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
            batch_predictions.append(predictions)
        darknet.free_batch_detections(batch_detections, self.batch_size)
        return images, batch_predictions

    def read_video_file(self) -> typ.Generator:
        video = cv2.VideoCapture(self.video_path)
        a = []
        while video.isOpened():
            frame_exist, frame = video.read()
            if frame_exist:
                if len(a) == BATCH_SIZE:
                    yield a
                    a = []
                else:
                    a.append(frame)
            else:
                print(f"Frame reading is finished!")
                break

    def batch_detection_process(self):
        a = [item for item in self.read_video_file()]
        predictions = []
        for i in a:
            images, detections, = self.batch_detection(
                self.network,
                i,
                self.class_names,
                self.class_colors,
            )
            predictions.append(detections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis video.')
    parser.add_argument(
        '-v',
        '--video',
        action="store",
        dest="video",
        required=True
    )
    # parser.add_argument(
    #     '-b',
    #     '--batchsize',
    #     action="store",
    #     type=int,
    #     dest="batchsize",
    #     default=settings.BATCH_SIZE,
    # )
    # parser.add_argument(
    #     '-t',
    #     '--trash',
    #     action="store",
    #     type=float,
    #     dest="trash",
    #     default=settings.IMAGE_THRESHOLD,
    # )
    parser.add_argument('-l', '--logfile', action="store", dest="logfile")
    arguments = parser.parse_args()
    conf_path = settings.CONFIG_PATH
    weight_path = settings.WEIGHT_PATH
    data_file_path = settings.DATA_FILE_PATH
    network, class_names, class_colors = darknet.load_network(
        conf_path,
        data_file_path,
        weight_path,
        batch_size=settings.BATCH_SIZE
    )
    VideoParser(
        settings.BATCH_SIZE,
        arguments.video,
        settings.IMAGE_THRESHOLD,
        network,
        class_names,
        class_colors
    ).batch_detection_process()
