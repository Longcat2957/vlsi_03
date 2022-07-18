import cv2
import numpy as np
from .core import Cv2PreProcessor

class RoiExtractor(object):
    '''
    Object Detection의 결과를 근거로 roi를 slice 한다(with deepcopy)
    size = (H, W)
    '''
    def __init__(self, size:tuple):
        self.cv2tensor = Cv2PreProcessor(size)

    def __call__(self, orig_img:np.ndarray, det1:np.ndarray):
        pts, out = self._slice(orig_img, det1)
        roi, ratios = self._resize(out)

        return pts, roi, ratios

    def _slice(self, orig_img:np.ndarray, det1:np.ndarray):
        pts, out = [], []
        for det in det1:
            x1, y1, x2, y2 = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = orig_img[x1:x2, y1:y2].copy()
            pts.append((x1,y1))
            out.append(roi)
        return pts, out

    def _resize(self, out:list):
        resized, ratios = [], []
        for o in out:
            padded, ratio = self.cv2tensor(o)
            resized.append(padded)
            ratios.append(ratio)
        return resized, ratios


if __name__ == '__main__':


    print('done')