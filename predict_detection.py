import os
import time
import cv2
import numpy as np
import pandas as pd
import yaml

from utils import *


def predict_detect_image(img, detector, cfg):
    conf_thresh = cfg["inference"]["confidence_threshold"]
    iou_thresh = cfg["inference"]["iou_threshold"]
    height, width = img.shape[:2]
    count_part_x = 1
    count_part_y = 1
    height_crop = int(height/count_part_y)
    width_crop = int(width/count_part_x)
    detection = []

    # cropping original image count_part_x*count_part_y parties and detection avery parties
    start_time_inference_frame_det = time.time()
    for part_x in range(count_part_x):
        for part_y in range(count_part_y):
            x0_crop = part_x * width_crop
            y0_crop = part_y * height_crop
            img_crop = img[y0_crop:y0_crop+height_crop,
                           x0_crop:x0_crop+width_crop]

            dets = detector([img_crop], conf_thresh=conf_thresh, iou_thresh=iou_thresh)[0]  # x_tl, y_tl, x_br, t_br
            dets[:, :4] = dets[:, :4].astype(np.int32)  # coordinates
            dets[:, 5] = dets[:, 5].astype(np.int32)    # id class

            dets[:, 0] = dets[:, 0] + x0_crop
            dets[:, 1] = dets[:, 1] + y0_crop
            dets[:, 2] = dets[:, 2] + x0_crop
            dets[:, 3] = dets[:, 3] + y0_crop
            detection.append(dets)
#     print(detection)
    duration_inference_frame_det = time.time() - start_time_inference_frame_det
    return np.vstack(detection), duration_inference_frame_det


def predict_detect_dir_images(cfg, detector):
    # Loading parameters model from config
    weights = cfg["model"]["weights"]
    class_names = cfg["model"]["class_names"]
    class_id_names = {class_id: class_name for class_id, class_name in enumerate(class_names)}
    input_height, input_width = map(int, cfg["model"]["input_size"].split(','))

    # Loading parameters for inference
    dir_img = cfg["inference"]["dir_img"]
    conf_tresh = cfg["inference"]["confidence_threshold"]
    iou_thresh = cfg["inference"]["iou_threshold"]
    is_create_xml = cfg["inference"]["create_xml"]

    predict_dir = os.path.join(dir_img, os.path.basename(weights).split(".")[0])
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    l_duration_inference_frame_det = []
    list_files = sorted(os.listdir(dir_img))

    df_predict = []
    output_df = os.path.basename(os.path.normpath(dir_img)) + "_detect.csv"
    for id_img, img_name in enumerate(list_files):
        df_row = {}
        path_image = os.path.join(dir_img, img_name)
        if os.path.isdir(path_image):
            continue
        ext_image = img_name.split(".")[-1]
        if ext_image.lower() in ['xml', 'txt']:
            continue
        if 'mask' in img_name:
            continue
        if 'overlay' in img_name:
            continue
        # if not "DJI" in img_name:
        #     continue

        print(f"{id_img}/{len(list_files)}", path_image)
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections, duration_inference_frame_det = predict_detect_image(img, detector, cfg)
        l_duration_inference_frame_det.append(duration_inference_frame_det)

        # draw result
        for id_bbox, bbox in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            df_row.update({"file_name": img_name})
            df_row.update({"file_path": path_image})
            df_row.update({"x_top_left": x1})
            df_row.update({"y_top_left": y1})
            df_row.update({"x_bottom_right": x2})
            df_row.update({"y_bottom_right": y2})
            df_row.update({"confidence_object": conf})
            df_row.update({"class_id": class_id})
            df_row.update({"class_name": class_id_names.get(class_id)})
            df_predict.append(df_row.copy())

            if is_create_xml:
                output_img = os.path.join(predict_dir, img_name)
                dst = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
                cv2.imwrite(output_img, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))

        classIds = [class_id_names.get(class_id) for class_id in detections[:, 5].astype(np.int32)]
        confidences = detections[:, 4].tolist()
        if is_create_xml:
            xml_name = img_name.replace(f".{ext_image}", ".xml")
            xml_file = os.path.join(predict_dir, xml_name)
            create_xml(xml_file,
                       path_image,
                       img.shape,
                       classIds=classIds,
                       confidences=confidences,
                       boxes=detections[:, :4].tolist())


        if (id_img % 100 == 0) or (id_img >= (len(list_files)-1)):
            print("save to csv", output_df)
            pd.DataFrame(df_predict).to_csv(output_df, index=False)

        # break
    return output_df