import cv2
import numpy as np

class RoiExtractor(object):
    def __call__(self, orig_img, fdet:np.ndarray):
        *position, _, _ = fdet
        position = np.array(position, dtype=np.int32)
        return orig_img[position[1]:position[3], position[0]:position[2]]


if __name__ == '__main__':


    print('done')