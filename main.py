import os
import argparse
import time
import cv2
import numpy as np
import pandas as pd
import yaml

from src.detector.YOLO import YOLOv5
from  src.classificator.Resnet import BinaryClassification
from predict_detection import *
from predict_classification import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='classification_trash')
    parser.add_argument("--cfg", help="path to yaml config", type=str,
                        default="./configs/config.yml")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    with open(cfg["configs"]["cfg_detection"], 'r') as ymlfile:
        cfg_detection = yaml.safe_load(ymlfile)
    with open(cfg["configs"]["cfg_classification"], 'r') as ymlfile:
        cfg_classification = yaml.safe_load(ymlfile)

    # Loading classification model - bad and good quality
    classificator_weights = cfg_classification["model"]["weights"]
    input_height_class, input_width_class = map(int, cfg_classification["model"]["input_size"].split(','))
    img_mean_r = cfg_classification["model"]["img_mean_r"]
    img_mean_g = cfg_classification["model"]["img_mean_g"]
    img_mean_b = cfg_classification["model"]["img_mean_b"]
    img_mean = (img_mean_r, img_mean_g, img_mean_b)
    img_std_r = cfg_classification["model"]["img_std_r"]
    img_std_g = cfg_classification["model"]["img_std_g"]
    img_std_b = cfg_classification["model"]["img_std_b"]
    img_std = (img_std_r, img_std_g, img_std_b)
    classificator = BinaryClassification(input_res=(input_height_class, input_width_class),
                                         weights=classificator_weights,
                                         batch_size=1,
                                         img_mean=img_mean,
                                         img_std=img_std)

    # Loading detection model - detect trash can
    detector_weights = cfg_detection["model"]["weights"]
    input_height_det, input_width_det = map(int, cfg_detection["model"]["input_size"].split(','))
    detector = YOLOv5(input_res=(input_height_det, input_width_det),
                      weights=detector_weights,
                      batch_size=1)

    dir_img = cfg["inference"]["dir_img"]
    cfg_detection["inference"].update({"dir_img": dir_img})
    cfg_classification["inference"].update({"dir_img": dir_img})

    output_df_detect = os.path.basename(os.path.normpath(dir_img)) + "_detection.csv"
    output_df_predict = os.path.basename(os.path.normpath(dir_img)) + "_predict.csv"
    output_df_class = predict_class_dir_images(cfg_classification, classificator)

    df_class: pd.DataFrame = pd.read_csv(output_df_class)
    df_detect = []
    df_predict = []
    for id_row, row in df_class.iterrows():
        df_row_detect = {}
        df_row_predict = {}
        path_image = row.get("file_path")
        img_name = row.get("file_name")
        predict_class_id = row.get("predict_class_id")   # 0 = good image, 1 = bad image

        print(f"{id_row}/{df_class.shape[0]}", path_image, predict_class_id)

        df_row_predict.update({"ID_img": img_name.split(".")[0]})

        if predict_class_id == 0:  # good image
            img = cv2.imread(path_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections, duration_inference_frame_det = predict_detect_image(img, detector, cfg_detection)
            if len(detections) == 0:
                df_row_predict.update({"class": 0})
            else:
                df_row_predict.update({"class": 1})
        else:
            detections = [[0, 0, 0, 0, 0, 0]]
            df_row_predict.update({"class": 2})

        df_predict.append(df_row_predict.copy())
        for id_bbox, bbox in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            df_row_detect.update({"file_name": img_name})
            df_row_detect.update({"file_path": path_image})
            df_row_detect.update({"x_top_left": x1})
            df_row_detect.update({"y_top_left": y1})
            df_row_detect.update({"x_bottom_right": x2})
            df_row_detect.update({"y_bottom_right": y2})
            df_row_detect.update({"confidence_object": conf})
            df_row_detect.update({"class_object": class_id})

            df_row_detect.update({"predict_class_id": predict_class_id})
            df_detect.append(df_row_detect.copy())

    print("save to csv", output_df_detect)
    pd.DataFrame(df_detect).to_csv(output_df_detect, index=False)

    print("save to csv", output_df_predict)
    pd.DataFrame(df_predict).to_csv(output_df_predict, index=False)
