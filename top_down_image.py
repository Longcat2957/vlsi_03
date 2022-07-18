import logging
import os
import sys
import cv2
import numpy as np

from utils.core.common import Cv2PreProcessor
from utils.core.yolo import YOLOv7Engine, YoloPostProcessor
from utils.ext import RoiExtractor
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
    def __init__(self, engine_path:str, infer_size:tuple):
        # *.trt, *.engine weight path
        self.engine_path = engine_path
        
        # preprcess, inference resolution
        self.infer_size = infer_size

        # unit's cores
        self.preprocessor = Cv2PreProcessor(self.infer_size)
        self.tensorrt_engine = YOLOv7Engine(self.engine_path, self.infer_size)
        self.postprocessor = YoloPostProcessor()

    @timer
    def infer(self, input):
        x, r = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x, r)

        return x

import cv2
from utils.io import LoadImage
from utils.core.common import Cv2PreProcessor
from utils.core.pose import TransPoseEngine, PoseHeatmapPostProcessor
from utils.obj import CocoPoseObjects
from utils.vis import CocoPoseVisualizer
from tools.benchmark import timer


class PoseInferenceBlock(object):
    def __init__(self, engine_path, joints, imgsz):
        self.preprocessor = Cv2PreProcessor(imgsz)
        self.tensorrt_engine = TransPoseEngine(engine_path, joints, imgsz)
        self.postprocessor = PoseHeatmapPostProcessor()

    def __call__(self, input):
        return self.infer(input)

    @timer
    def infer(self, input):
        x, r = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x, r)
        return x


if __name__ == '__main__':
    yolo_engine_path = './model/yolov7_480x640.engine'
    pose_engine_path = './model/transpose-h.engine'
    test_img_path = './input/sample_image_0.jpeg'

    orig_img = cv2.imread(test_img_path)

    yolo_block = YOLOv7_Inference_Block(yolo_engine_path, (480, 640))
    pose_block = PoseInferenceBlock(pose_engine_path, 17, (256, 192))
    
    dets = yolo_block.infer(orig_img)
    limitations = [0]
    yoloobjects = YoloObjects(limitations)
    fdets = yoloobjects(dets)

    yolo_vis = YoloVisualizer()
    yolo_img = yolo_vis(orig_img, fdets)

    my_roi = RoiExtractor()
    for human in fdets:
        # fdets format = (x1, x2, y1, y2, p1, p2)
        position = my_roi(orig_img, human)
        preds, _ = pose_block(position)
        pose_obj = CocoPoseObjects(True)    # with lines
        pdets, lines = pose_obj(preds)
        pose_vis = CocoPoseVisualizer()
        pose = pose_vis(position, pdets, lines)
        cv2.imshow('pose', pose)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print('done')