import os
import time
import cv2
import numpy as np
import pandas as pd
import yaml

from utils import *


def predict_class_image(img, classificator):
    height, width = img.shape[:2]
    count_part_x = 2
    count_part_y = 2
    height_crop = int(height / count_part_y)
    width_crop = int(width / count_part_x)
    start_time_inference_frame_class = time.time()

    predict_full = classificator([img])[0]
    x_0, x_1, x_2 = predict_full[0], predict_full[1], predict_full[2]
    print('full_predict', predict_full, (-x_0+x_1+x_2+1)/2)
    # if predict_full.max() > 0.9:
    if (-x_0+x_1+x_2+1)/2 > 0.9:
        duration_inference_frame_class = time.time() - start_time_inference_frame_class
        return predict_full, duration_inference_frame_class
    if (-x_0+x_1+x_2+1)/2 < 0.2:
        duration_inference_frame_class = time.time() - start_time_inference_frame_class
        return predict_full, duration_inference_frame_class
    # duration_inference_frame_class = time.time() - start_time_inference_frame_class
    # return predict_full, duration_inference_frame_class
    predicts = []
    for part_x in range(count_part_x):
        for part_y in range(count_part_y):
            x0_crop = part_x * width_crop
            y0_crop = part_y * height_crop
            img_crop = img[y0_crop:y0_crop + height_crop,
                       x0_crop:x0_crop + width_crop]
            crop_predict = classificator([img_crop])[0]
            predicts.append(crop_predict)
    predicts = np.array(predicts)
    print('crop_predict', predicts, (-predicts[:, 0]+predicts[:, 1]+predicts[:, 2]+1)/2 )
    print('median_predict', np.median(predicts, axis=0), np.median((-predicts[:, 0]+predicts[:, 1]+predicts[:, 2]+1)/2))
    duration_inference_frame_class = time.time() - start_time_inference_frame_class
    predict_mean = np.median(predicts, axis=0)
    predict = predict_mean
    # if predict_mean.max() > predict_full.max():
    #     predict = predict_mean
    # else:
    #     predict = predict_full
    print("predict", predict)
    return predict, duration_inference_frame_class


def predict_class_dir_images(cfg, classificator):
    # Loading parameters model from config
    weights = cfg["model"]["weights"]
    class_names = cfg["model"]["class_names"]
    class_id_names = {class_id: class_name for class_id, class_name in enumerate(class_names)}
    input_height, input_width = map(int, cfg["model"]["input_size"].split(','))

    # Loading parameters for inference
    dir_img = cfg["inference"]["dir_img"]
    conf_tresh = cfg["inference"]["confidence_threshold"]

    l_duration_inference_frame_class = []
    list_files = sorted(os.listdir(dir_img))

    df_predict = []
    output_df = os.path.basename(os.path.normpath(dir_img)) + "_class.csv"
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
        class_predict, duration_inference_frame_class = predict_class_image(img, classificator)
        l_duration_inference_frame_class.append(duration_inference_frame_class)

        df_row.update({"file_name": img_name})
        df_row.update({"file_path": path_image})
        for class_id, class_name in enumerate(class_names):
            df_row.update({f"{class_name}": class_predict[class_id]})

        # predict_class_id = class_predict.argmax()
        score_class_id = (-class_predict[0]+class_predict[1]+class_predict[2]+1)/2
        print(f"{path_image}, score_class_id: {score_class_id}")
        if score_class_id > 0.5:
            predict_class_id = 1
        else:
            predict_class_id = 0

        df_row.update({"predict_class_name": class_id_names.get(predict_class_id)})
        df_row.update({"predict_class_id": predict_class_id})
        df_row.update({"score_class_id": score_class_id})
        df_predict.append(df_row.copy())

        if (id_img % 100 == 0) or (id_img >= (len(list_files)-1)):
            print("save to csv", output_df)
            pd.DataFrame(df_predict).to_csv(output_df, index=False)

        # break
    return  output_df