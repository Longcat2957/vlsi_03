import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
from .obj import YoloObjects

class Visualizer(object):
    """
    Visualizer base-class
    ======================    
    """
    def __init__(self):
        self.point_thickness = 4
        self.point_color = (255, 0, 0)

        self.line_thickness = 1
        self.line_color = (0, 255, 0)

        self.box_thickness = 2
        self.box_color = (0, 0, 255)
        
        self.chr_font = cv2.FONT_HERSHEY_PLAIN
        self.chr_thickness = 2
        self.chr_color = (0, 0, 0)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
        
    def _draw_point(self, orig_img, point:tuple):
        orig_img = cv2.line(orig_img, point, point, self.point_color, self.point_thickness)
        return orig_img

    def _draw_line(self, orig_img, start:tuple, end:tuple):
        orig_img = cv2.line(orig_img, start, end, self.line_color, self.line_thickness)
        return orig_img

    def _draw_boxes(self, orig_img, boxes):
        pt1, pt2 = (boxes[0], boxes[1]), (boxes[2], boxes[3])
        out = cv2.rectangle(orig_img, pt1, pt2, self.box_color, self.box_thickness)
        return out

    def _write_text(self, orig_img, position:tuple, message:str):
        orig_img = cv2.putText(orig_img, message, position, self.chr_font, 1, self.chr_color, self.chr_thickness)
        return orig_img



class YoloVisualizer(Visualizer):

    def __init__(self):
        super(YoloVisualizer, self).__init__()

    def __call__(self, orig_img, dets):
        for det in dets:
            *boxes, p, c = det
            boxes = np.array(boxes)
            rboxes = boxes.astype(np.int32)
            orig_img = self._draw_boxes(orig_img, rboxes)
        
        return orig_img



if __name__ == '__main__':
    limitations = [0, 1]
    yoloobjects = YoloObjects(limitations)
    dets_sample = [[100, 200, 300, 400, 0.9, 0], [100, 200, 300, 400, 0.9, 10]]
    dets_sample = np.array(dets_sample)
    yoloobjects.load_dets(dets_sample)
    dets = yoloobjects(dets_sample)
    print(dets.shape)

    myyolovis = YoloVisualizer((480, 640), (576, 768))
    test_image_path = '../input/test1.jpeg'
    test_img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

    test_img = myyolovis(test_img, dets)

    cv2.imshow('image', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()