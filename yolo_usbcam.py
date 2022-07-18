import logging
import os
import sys
import cv2
import numpy as np

from utils.core.common import Cv2PreProcessor
from utils.core.yolo import YOLOv7Engine, YoloPostProcessor
from utils.vis import YoloVisualizer
from utils.obj import YoloObjects
from tools.benchmark import timer

class YOLOv7_Inference_Block(object):
    """
    Object Detsction Inference Block
    ======================
    임의의 cv2 인풋을 받아 object detection 결과를 리턴합니다.

    ----------------------
    ATTRIBUTS //
    engine_path : *.trt, *.engine의 directory
    infer_size : 추론 모델의 입력 텐서 shape
    conf_score : confidence threshold(default = 0.45)
    nms_thr : non-maximum-supression threshold(default = 0.1)

    ----------------------
    METHOD //
    infer(orig_img) -> dets:np.ndarray : 추론을 수행한다. (with rescale)

    """
    def __init__(self, engine_path:str, infer_size:tuple, conf_scores:float, nms_thr:float, verbose:bool):
        # *.trt, *.engine weight path
        self.engine_path = engine_path
        
        # preprcess, inference resolution
        self.infer_size = infer_size
        
        # postproces parameters
        self.conf_scores = conf_scores
        self.nms_thr = nms_thr

        # logging
        self.verbose = verbose

        # unit's cores
        self.preprocessor = Cv2PreProcessor(self.infer_size)
        self.tensorrt_engine = YOLOv7Engine(self.engine_path, self.infer_size)
        self.postprocessor = YoloPostProcessor(self.conf_scores, self.nms_thr)

    def __call__(self, input):
        return self.infer(input)

    #@timer
    def infer(self, input):
        x, r = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x, r)

        return x


if __name__ == '__main__':
    
    # config
    engine_path = './model/yolov7_480x640.engine'
    video_idx = 0
    inference_block = YOLOv7_Inference_Block(engine_path, (480, 640), 0.45, 0.1, False)
    
    yolo_limitation = [0]   # class 0 is human
    yolo_obj = YoloObjects(yolo_limitation)
    yolo_visualizer = YoloVisualizer()

    #################################
    cam = cv2.VideoCapture(video_idx)

    WIDTH = 640
    HEIGHT = 480
    FRAMERATE = 30

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, FRAMERATE)

    while cv2.waitKey(1) < 1:

        grabbed, orig_img = cam.read()

        if not grabbed:
            exit()
        
        dets = inference_block(orig_img)
        dets = yolo_obj(dets)
        yolo_img = yolo_visualizer(orig_img, dets)
        cv2.imshow('test', yolo_img)
