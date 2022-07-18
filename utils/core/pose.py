import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

class PoseBaseEngine(object):
    """
    """
    def __init__(self, engine_path, joints:int, imgsz:tuple):
        self.imgsz = imgsz
        self.joints = joints
        
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
        predictions = data[0]
        predictions = np.reshape(predictions, (self.joints, self.imgsz[0]//4, -1))
        return predictions


class TransPoseEngine(PoseBaseEngine):
    def __init__(self, engine_path:str, joints:int, imgsz:tuple):
        super(TransPoseEngine, self).__init__(engine_path, joints, imgsz)



class PoseHeatmapPostProcessor(object):
    """
    Heatmap --> preds, maxvals
    """
    def __call__(self, heatmap, ratio):
        preds, maxvals = self._getmaxpredicts(heatmap)
        preds /= ratio
        preds *= 4.0
        return preds.astype(np.int32), maxvals

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