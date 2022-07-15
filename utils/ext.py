import cv2
import numpy as np
from core import Cv2PreProcessor

class RoiExtractor(object):
    '''
    Object Detection의 결과를 근거로 roi를 slice 한다(with deepcopy)
    '''
    def __init__(self):
        self.cv2tensor = Cv2PreProcessor((224, 224))

    def __call__(self, orig_img:np.ndarray, det1:np.ndarray):
        out = self._slice(orig_img, det1)
        final = self._resize(out)

        if len(final) == 1:
            return final[0]
        return final

    def _slice(self, orig_img:np.ndarray, det1:np.ndarray):
        out = []
        for det in det1:
            x1, y1, x2, y2, _, _ = det
            roi = orig_img[x1:x2, y1:y2].copy()
            out.append(roi)
    
    def _resize(self, out:list):
        final = []
        for o in out:
            final.append(self.cv2tensor(o))
        return final


if __name__ == '__main__':


    print('done')