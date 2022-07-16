#import tensorrt as trt
#import pycuda.autoinit
#import pycuda.driver as cuda
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


class YoloBaseEngine(object):
    """
    YOLOv5, YOLOv6, YOLOv7 compatible base engine class
    https://github.com/Linaom1214/tensorrt-python/blob/main/utils/utils.py
    
    __init__ : engine_path:str, imgsz(=neural network input shape):tuple
    __call__ : np.ndarray(C, H, W) --> np.ndarray(C, H, W)
    """
    def __init__(self, engine_path, imgsz=(480, 640)):
        self.imgsz = imgsz
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

    def __call__(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
        return predictions

class YoloPostProcessor(object):
    """
    TensorRT 또는 ONNX-Runtime의 출력 텐서를 nms+multiclass_nms를 통해 후처리합니다.

    __init__ : conf_scores:float, nms_thr:float
    __call__ : np.ndarray(B, _, 85) --> np.ndarray(B, _, 6)
    """
    def __init__(self, conf_scores, nms_thr):
        self.conf_scores = conf_scores
        self.nms_thr = nms_thr

    def __call__(self, predictions, ratio):
        return self.postprocess(predictions, ratio)

    def postprocess(self, predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores)
        return dets

    def nms(self, boxes, scores):
        '''
        Non-Max-Suppression
        
        '''
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
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= self.nms_thr)[0]
            order = order[inds + 1]
        
        return keep
    
    def multiclass_nms(self, boxes, scores):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.conf_scores
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)


class PoseHeatmapPostProcessor(object):
    """
    Heatmap --> preds, maxvals
    """
    def __call__(self, heatmap):
        return self._getmaxpredicts(heatmap)

    def _getmaxpredicts(self, hmap):
        num_joints, H, W = hmap.shape
        hmap_flatten = hmap.reshape((num_joints, H*W)) 
        idx = np.argmax(hmap_flatten, 1).reshape(num_joints, 1)
        maxvals = np.amax(hmap_flatten, 1).reshape(num_joints, 1)
        
        preds = np.tile(idx,(1,2)).astype(np.float32)

        preds[:,0] = (preds[:,0]) % W
        preds[:,1] = np.floor((preds[:, 1])/W)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1,2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask     #pred= pred* pres_mask
        return preds, maxvals

class YOLOv7Engine(YoloBaseEngine):
    """
    Yolov7 engine
    """
    def __init__(self, engine_path, imgsz=(480, 640)):
        super(YOLOv7Engine, self).__init__(engine_path)
        self.imgsz = imgsz
        self.n_classes = 80
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
    test_image_path = '../input/test1.jpeg'

    test_img = cv2.imread(test_image_path)
    print()
    print(f'입력 이미지의 shape = {test_img.shape}')
    ############################################################### << debug
    
    # test PreProcessor class
    test_pre_processor = Cv2PreProcessor((480, 640))
    dummy_input, ratio = test_pre_processor._preproc(test_img)
    dummy_input, ratio = test_pre_processor(test_img)
    print(f'PreProcessor 클래스에 의해 (C, H, W)로 변환 / 변환비율 = {dummy_input.shape} / ratio={ratio}')
    ############################################################### << debug

    # test YOLOv7_engine class
    test_yolov7_engine = YOLOv7Engine(test_engine_path, imgsz=(480, 640))
    dummy_output = test_yolov7_engine(dummy_input)
    print(f'TensorRT에 의한 추론 결과, reshape 포함 = {dummy_output.shape}')
    ############################################################### << debug

    # test PostProcessor class
    test_post_processor = YoloPostProcessor(0.45 ,0.1)
    dummy_output_2 = test_post_processor(dummy_output, ratio)
    print(f'Multi-Class NMS에 의한 중복된 박스 제거 = {dummy_output_2.shape}')
    
