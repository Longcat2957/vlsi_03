import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2


class Visualizer(object):
    """
    YOLO-series Visualizer
    ======================    
    """
    def __init__(self, model_shape:tuple ,output_size:tuple, frame:int, format:str):
        self.tensor_shape = model_shape
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        
        self.frame = frame
        self.format = format
        self.ratio = self._get_ratio()

    def set_size(self, size):
        if isinstance(size, int):
            self.output_size = (size, size)
        elif isinstance(size, tuple):
            self.output_size = size
        return f'output_size = {self.output_size}'

    def set_frame(self, frame):
        if isinstance(frame, int):
            self.frame = frame
        return f'frame = {self.frame}'
    
    def __call__(self, orig_img, dets):

        pass

    def _draw_point(self, point:tuple):

        pass

    def _draw_line(self, start:tuple, end:tuple):

        pass

    def _draw_boxes(self, boxes):

        pass

    def _write_text(self, position:tuple, message:str):

        pass

    def _get_ratio(self):
        if self.tensor_shape == self.output_size:
            return True
        else:
            return (self.output_size[0] / self.tensor_shape[0]) , (self.output_size[1], self.tensor_shape[1])

    def _vis(self, orig_img, dets):

        pass

if __name__ == '__main__':

    print('done')