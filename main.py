import os
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import yaml

from src.detector.YOLO import YOLOv5
from predict_detection import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detection Golf')
    parser.add_argument("--cfg", help="path to yaml config", type=str, default="./config/config.yml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Loading parameters model from config
    weights = cfg["model"]["weights"]
    input_height, input_width = map(int, cfg["model"]["input_size"].split(','))

    print("Initialize NN")
    detector = YOLOv5(input_res=(input_height, input_width),
                      weights=weights,
                      batch_size=1)
    print("Initialize NN: success")

    l_duration_inference_frame_det = predict_dir_images(cfg, detector)

    print("duration_inference_frame_det", np.mean(l_duration_inference_frame_det[:]))
