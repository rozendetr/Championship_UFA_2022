import cv2
import onnxruntime as ort
import time
import numpy as np
import os
from .utils import *
import albumentations as A


# onnx model
class BinaryClassification:
    def __init__(self,
                 weights: str = None,
                 input_res: tuple = (256, 256),
                 batch_size: int = 1,
                 img_mean=(121.22450319, 113.15888359, 110.76126834),
                 img_std=(64.79503432, 65.45959436, 66.49727735)
                 ):
        super().__init__()
        self.weights = weights
        self.input_res = input_res
        self.batch_size = batch_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.ort_session, self.input_name = self._init_session_(self.weights)
        self.normalize = A.Compose([A.Normalize(mean=self.img_mean, std=self.img_std)])

    def _init_session_(self, path_onnx_model: str):
        ort_session = None
        input_name = None
        if os.path.isfile(path_onnx_model):
            try:
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CUDAExecutionProvider'])
            except:
                ort_session = ort.InferenceSession(path_onnx_model, providers=['CPUExecutionProvider'])
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, imgs: list):
        imgs_input = []
        for img in imgs:
            img_input, ratio, (dw, dh) = letterbox(img,
                                                   self.input_res,
                                                   auto=False,
                                                   scaleFill=False,
                                                   scaleup=True,
                                                   stride=32)
            f_img_input = generate_FT(img_input)
            img_input = self.normalize(image=img_input)["image"]
            img_input = np.dstack((img_input, f_img_input)).astype(np.float32)
            # img_input = np.stack([f_img_input, f_img_input, f_img_input], axis=2).astype(np.float32)
            img_input = img_input.transpose(2, 0, 1)
            img_input = np.expand_dims(img_input, axis=0)
            imgs_input.append(img_input)
        return imgs_input

    def postprocessing(self, predictions, imgs):
        assert len(predictions) == len(imgs), f"Size prediction {len(predictions)} not equal size images {len(imgs)}"
        soft_max_predictions = softmax(np.array(predictions))
        return soft_max_predictions

    def __call__(self, imgs):
        if not self.ort_session:
            return False

        if self.batch_size == 1:
            preds = []
            for img in imgs:
                onnx_result = self.ort_session.run([],
                                                   {self.input_name: self.preprocessing([img])[0]})
                pred = onnx_result[0]
                pred = self.postprocessing(predictions=pred, imgs=[img])
                preds.append(pred[0])
            return preds

        else:
            input_imgs = self.preprocessing(imgs)
            input_imgs = np.concatenate(input_imgs, axis=0)
            onnx_result = self.ort_session.run([], {self.input_name: input_imgs})
            pred = onnx_result[0]
            pred = self.postprocessing(predictions=pred, imgs=imgs)
        return pred