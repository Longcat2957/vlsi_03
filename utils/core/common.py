import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

class Cv2PreProcessor(object):
    """
    Cv2 형식의 np.ndarray를 torch type의 np.ndarray로 변환합니다.
    TensorRT 호환성을 위해 np.ascontiguousarray의 결과를 리턴합니다.

    __init__ : imgsz:tuple
    __call__ : orig_img:np.ndarray(H, W, C) --> Tensor_input:np.ndarray(C, H, W)
    """
    def __init__(self, imgsz:tuple):
        self.imgsz = imgsz  # 목표로 하는 크기
        self.target_h, self.target_w = self.imgsz[0], self.imgsz[1]
        self.area = self.imgsz[0] * self.imgsz[1] # 면적의 크기, 보간법을 서로 다르게 한다.

    def __call__(self, raw_input):
        return  self._preproc(raw_input)
    
    def _preproc(self, image):
        if len(image.shape) == 3:
            padded_img = np.ones((self.target_h, self.target_w, 3)) * 114.0     # color(3 channels)
        else:
            padded_img = np.ones(self.imgsz) * 114.0                            # greyscale

        img = np.array(image)
        r = min(self.target_h / img.shape[0], self.target_w / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0

        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r


class HeatmapResizer(object):

    def __init__(self, imgsz:tuple):
        self.imgsz = imgsz
        self.target_h, self.target_w = self.imgsz[0], self.imgsz[1]
    
    def __call__(self, heatmap):
        return self._resize(heatmap)

    def _resize(self, heatmap):
        # ex. 17x64x48 --> 17x56x56
        hmap = np.array(heatmap) # copy
        hmap = hmap.transpose(1, 2, 0)
        rhmap = cv2.resize(hmap, self.imgsz)
        return rhmap.transpose(2, 0, 1)

if __name__ == '__main__':
    random = np.ndarray((17, 64, 48), dtype=np.float32)
    myhrs = HeatmapResizer((56, 56))
    out = myhrs.__call__(random)
    print(f'out = {out.shape}')