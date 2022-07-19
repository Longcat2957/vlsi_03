import cv2
import numpy as np
from copy import deepcopy

class RoiExtractor(object):
    def __call__(self, orig_img, fdet:np.ndarray):
        rois = []
        for fd in fdet:
            *position, _, _ = fd
            position = np.array(position, dtype=np.int32)
            roi = deepcopy(orig_img[position[1]:position[3], position[0]:position[2]])
            rois.append(roi)
        return rois

class RoiExtractor_nc(object):
    def __call__(self, orig_img, fdet:np.ndarray):
        rois = []
        for fd in fdet:
            *position, _, _ = fd
            position = np.array(position, dtype=np.int32)
            roi = orig_img[position[1]:position[3], position[0]:position[2]]
            rois.append(roi)
        return rois

if __name__ == '__main__':


    print('done')