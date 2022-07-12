import logging
import os
import sys
import cv2
import numpy as np

from core import PreProcessor, YOLOv7_engine, PostProcessor, Visualizer

class YOLOv7_trt_unit(object):
    """
    Unifed Inference Class
    """
    def __init__(self, engine_path:str, infer_size:tuple, vis_size:tuple, obj_class_list:list, conf_scores:float, nms_thr:float, verbose:bool):
        # *.trt, *.engine weight path
        self.engine_path = engine_path
        
        # preprcess, inference resolution
        self.infer_size = infer_size
        
        # output resolution
        self.vis_size = vis_size

        # postprocess parameters (1)
        self.obj_class_list = obj_class_list
        self.obj_class_num = len(obj_class_list)
        
        # postproces parameters (2)
        self.conf_scores = conf_scores
        self.nms_thr = nms_thr

        # logging
        self.verbose = verbose

        # unit's cores
        self.preprocessor = PreProcessor(self.infer_size)
        self.tensorrt_engine = YOLOv7_engine(self.engine_path, self.infer_size)
        self.postprocessor = PostProcessor(self.conf_scores, self.nms_thr)
        self.visualizer = Visualizer(self.vis_size)

    def infer(self, input):
        x = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x)
        x = self.visualizer(x)
        return x