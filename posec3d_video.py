import logging
import os
import sys
import cv2
import numpy as np
import argparse

import time

from utils.core.eco import ECOEngine, EcoPostProcessor
from utils.ext import RoiExtractor
from utils.obj import HeatmapSequenceObjects, VideoSequenceObjects

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
    def __init__(self, engine_path:str, infer_size:tuple):
        # *.trt, *.engine weight path
        self.engine_path = engine_path
        
        # preprcess, inference resolution
        self.infer_size = infer_size

        # unit's cores
        self.preprocessor = Cv2PreProcessor(self.infer_size)
        self.tensorrt_engine = YOLOv7Engine(self.engine_path, self.infer_size)
        self.postprocessor = YoloPostProcessor()

    def __call__(self, input):
        return self.infer(input)

    #@timer
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

    #@timer
    def infer(self, input):
        x, r = self.preprocessor(input)
        hmap = self.tensorrt_engine(x)
        dets = self.postprocessor(hmap, r)
        return dets[0], dets[1], hmap

from utils.core.posec3d import PoseC3dEngine, ArPostProcessor
from utils.core.common import HeatmapResizer

class PoseC3dInferenceBLock(object):
    def __init__(self, engine_path, classes, frames, imgsz):
        self.tensorrt_engine = PoseC3dEngine(engine_path, classes, frames, imgsz)
        self.arpp = ArPostProcessor(5)

    def __call__(self, input):
        return self.infer(input)
    

    def infer(self, input):
        out = self.tensorrt_engine(input)
        pred = self.arpp(out)
        return pred


from utils.ext import RoiExtractor, RoiExtractor_nc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input/test21.mp4')
    opt = parser.parse_args()
    ##################################################
    # YOLO
    yolo_engine_path = './model/yolov7_480x640.engine'
    yolo_block = YOLOv7_Inference_Block(yolo_engine_path, (480, 640))
    yolo_limit = [0]

    # YOLO obj
    yolo_obj = YoloObjects(yolo_limit)
    
    # YOLO vis
    yolo_vis = YoloVisualizer()
    
    ###################################################
    # Pose-estimation
    pe_engine_path = './model/transpose-h.engine'
    pe_block = PoseInferenceBlock(pe_engine_path, 17, (256, 192))
    pe_pp = PoseHeatmapPostProcessor()
    pose_obj = CocoPoseObjects(True)
    pose_vis = CocoPoseVisualizer()

    ##################################################
    # PoseC3D
    ECO_engine_path = './model/posec3d_v1.engine'
    eco_block = PoseC3dInferenceBLock(ECO_engine_path, 17, 32, (56, 56))

    ###################################################
    # video
    video_file_path = opt.input
    cam = cv2.VideoCapture(video_file_path)
    hrresize = HeatmapResizer((56, 56))
    ###################################################
    roi_ext = RoiExtractor_nc()
    vid_seq = HeatmapSequenceObjects()   # set_default

    t0 = time.time()
    while cv2.waitKey(1) < 1:
        grabbed, orig_img = cam.read()
        if not grabbed:
            exit()
        
        dets = yolo_block(orig_img)
        dets = yolo_obj(dets)
        yolo_img = yolo_vis(orig_img, dets)

        if dets is not None:
            rois = roi_ext(orig_img, dets)
            
            for roi in rois:
                ppreds, _, hmap = pe_block.infer(roi)
                pdets, lines = pose_obj(ppreds)
                pose_img = pose_vis(roi, pdets, lines)  #
                
                vid_seq.push(hmap)
        
        cv2.imshow('bbox', yolo_img)

        if time.time() - t0 > 1.80:
            out = vid_seq.pull()
            vid_seq.clear()
            print(out.shape)


            
        

    cv2.destroyAllWindows()
    print('done')