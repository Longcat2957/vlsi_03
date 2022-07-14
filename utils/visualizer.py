import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

class YoloObjects(object):
    '''
    YoloObjects Class
    =================
    limitation:list 에 근거하여 목표로 한 클래스를 제외하고 리턴한다.
    '''
    def __init__(self, limitation=None):
        self.org_dets = None    # type : numpy.ndarray | example : (N, 6)
        self.limitation = limitation
        self.dets = None

    def __call__(self, dets):
        self.load_dets(dets)
        self._discard()
        return self.dets

    def load_dets(self, dets):
        self.org_dets = dets

    def _discard(self):
        n, _ = self.org_dets.shape
        class_array = self.org_dets[:, 5].flatten()
        valid = np.array([x in self.limitation for x in class_array])
        self.dets = self.org_dets[valid]


class Visualizer(object):
    """
    YOLO-series Visualizer
    ======================    
    """
    def __init__(self, model_shape:tuple ,output_size:tuple):
        self.tensor_shape = model_shape
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        
        self.ratio = self._get_ratio()
        self.color = (0, 0, 255)
        self.thickness = 3
        self.font = cv2.FONT_HERSHEY_PLAIN

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
        raise NotImplementedError
        
    def _draw_point(self, orig_img, point:tuple):
        orig_img = cv2.line(orig_img, point, point, self.color, self.thickness)
        return orig_img

    def _draw_line(self, orig_img, start:tuple, end:tuple):
        orig_img = cv2.line(orig_img, start, end, self.color, self.thickness)
        return orig_img

    def _draw_boxes(self, orig_img, boxes):
        pt1, pt2 = (boxes[0], boxes[1]), (boxes[2], boxes[3])
        out = cv2.rectangle(orig_img, pt1, pt2, self.color, self.thickness)
        return out

    def _write_text(self, orig_img, position:tuple, message:str):
        orig_img = cv2.putText(orig_img, message, position, self.font, 1, self.color, self.thickness)
        return orig_img

    def _get_ratio(self):
        if self.tensor_shape == self.output_size:
            return True
        else:
            return (self.output_size[0] / self.tensor_shape[0] , self.output_size[1] / self.tensor_shape[1])


class YoloVisualizer(Visualizer):

    def __init__(self, model_shape, model_output):
        super(YoloVisualizer, self).__init__(model_shape, model_output)

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