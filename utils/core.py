import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

class PreProcessor(object):
    """
    
    """
    def __init__(self, imgsz):
        self.imgsz = imgsz

    def __call__(self, raw_input):


        return


class BaseEngine(object):
    """
    YOLOv5, YOLOv6, YOLOv7 compatible base engine class
    https://github.com/Linaom1214/tensorrt-python/blob/main/utils/utils.py
    """
    def __init__(self, engine_path, imgsz=(480, 640)):
        self.imgsz = imgsz
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_name = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()

        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def __call__(self, img:np.ndarray) -> list:
        """
        type of img : numpy.ndarray
        """
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        # run inference
        self.context.execute_async_v2(
            bindings = self.bindings,
            stream_handle = self.stream.handle
        )

        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # synchronize stream
        self.stream.synchronize()
        
        data = [out['host'] for out in self.outputs]
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
        return predictions

class PostProcessor(object):
    """
    
    
    """
    def __init__(self, conf_scores, nms_thr):
        self.conf_socres = conf_scores
        self.nms_thr = nms_thr

    def __call__(self, predictions):
        return self._postprocess(predictions)

    def _postprocess(self, predictions):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        dets = self._multiclass_nms(boxes_xyxy, scores)
        return dets

    def _nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.maximum(x2[i], x2[order[1:]])
            yy2 = np.maximum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:] - inter])
            inds = np.where(ovr <= self.nms_thr)[0]
            order = order[inds + 1]
        return keep
    
    def _multiclass_nms(self, boxes, scores):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.conf_socres
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self._nms(valid_boxes, valid_scores, self.nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

class Visualizer(object):
    """
    
    """
    def __init__(self):

        pass

    def __call__(self):

        pass


class YOLOv7_engine(BaseEngine):
    """
    
    """
    def __init__(self, engine_path, imgsz):
        super(YOLOv7_engine, self).__init__(engine_path, imgsz)
        self.imgsz = imgsz
        self.n_classes = 80 # COCO Detection
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


if __name__ == '__main__':
    """
    Debug PreProcessor, BaseEngine, PostProcessor, Visualizer
    
    """
    test_engine_path = '../model/yolov7_480x640.engine'
    test_image_path = '../src/test1.jpeg'                       # 디버깅용 피클을 만들어야 할듯 ㅠ

    # test YOLOv7_engine class
    dummy_input = np.ndarray((1, 3, 480, 640), dtype=np.float16)
    test_yolov7_engine = YOLOv7_engine(engine_path = test_engine_path, imgsz=(480, 640))
    dummy_output = test_yolov7_engine(dummy_input)
    print(dummy_output.shape)

    # test PostProcessor class
    test_post_processor = PostProcessor(0.45 ,0.1)
    dummy_output_2 = test_post_processor(dummy_output)
    print(dummy_output_2.shape)
    print("Test Complete !")