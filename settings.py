# -*- coding: utf-8 -*-
import os

# YOLO4 compiled Files
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/home/ubuntu/Yolo_4_26_10_2020/darknet/cfg/yolov4.cfg")
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/home/ubuntu/Yolo_4_26_10_2020/darknet/yolov4.weights")
DATA_FILE_PATH = os.environ.get("META_PATH", "/home/ubuntu/Yolo_4_26_10_2020/darknet/cfg/coco.data")
DARKNET_PATH = os.environ.get("META_PATH", "/home/ubuntu/Yolo_4_26_10_2020/darknet/")
#DARKNET_LIB_PYTHON_PATH="/home/tito/SpilnaSprava/Mollengo/YOLOBatchProcessing/darknet/darknet.py"
IMAGE_THRESHOLD = 0.25

# Batch size
BATCH_SIZE = 4

# Logging
LOGGING_LEVEL = "INFO"
LOGGING_FILE = "/var/log/video_proc.log"
#LOGGING_FORMATTER="%(asctime)s - %(name)s - %(levelname)s - %(message)s"