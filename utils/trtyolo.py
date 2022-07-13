import logging
import os
import sys
import cv2
import numpy as np

from core import YoloPreProcessor, YOLOv7Engine, YoloPostProcessor

class YOLOv7_trt_unit(object):
    """
    Unifed Inference Class
    ======================
    임의의 cv2 인풋을 받아 object detection 결과를 리턴합니다.
    """
    def __init__(self, engine_path:str, infer_size:tuple, obj_class_list:list, conf_scores:float, nms_thr:float, verbose:bool):
        # *.trt, *.engine weight path
        self.engine_path = engine_path
        
        # preprcess, inference resolution
        self.infer_size = infer_size

        # postprocess parameters (1)
        self.obj_class_list = obj_class_list
        self.obj_class_num = len(obj_class_list)
        
        # postproces parameters (2)
        self.conf_scores = conf_scores
        self.nms_thr = nms_thr

        # logging
        self.verbose = verbose

        # unit's cores
        self.preprocessor = YoloPreProcessor(self.infer_size)
        self.tensorrt_engine = YOLOv7Engine(self.engine_path, self.infer_size)
        self.postprocessor = YoloPostProcessor(self.conf_scores, self.nms_thr)

    def infer(self, input):
        x = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x)

        return x