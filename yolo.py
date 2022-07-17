import logging
import os
import sys
import cv2
import numpy as np

from utils.core import Cv2PreProcessor, YOLOv7Engine, YoloPostProcessor
from utils.vis import YoloVisualizer
from utils.obj import YoloObjects

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

    def infer(self, input):
        x, r = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x, r)

        return x

if __name__ == '__main__':
    test_engine_path = './model/yolov7_480x640.engine'
    test_image_path = './input/test1.jpeg'

    test_img = cv2.imread(test_image_path)

    myunit = YOLOv7_Inference_Block(test_engine_path, (480, 640), 0.45, 0.1, False)
    dets = myunit.infer(test_img)
    

    limitations = [x for x in range(80)]
    yoloobjects = YoloObjects(limitations)
    fdets = yoloobjects(dets)

    myyolovis = YoloVisualizer()
    
    final = myyolovis(test_img, fdets)
    cv2.imshow('image', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('done')