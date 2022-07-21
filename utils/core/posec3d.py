from re import S
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

class SkeletonBasedActionRecognitionBE(object):
    """
    """
    def __init__(self, engine_path, classes:int, frames:int ,imgsz:tuple):
        self.frames = frames
        self.imgsz = imgsz
        self.classes = classes
        
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

    def __call__(self, sequence):
        self.inputs[0]['host'] = np.ravel(sequence)
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
        print(data[0].shape)
        predictions = np.reshape(data, (1, self.classes))[0]
        return predictions


class PoseC3dEngine(SkeletonBasedActionRecognitionBE):
    '''
    Tensor-RT ECO inference(UCF-101)
    '''
    def __init__(self, engine_path:str, classes:int, frames:int, imgsz:tuple):
        super(PoseC3dEngine, self).__init__(engine_path, classes, frames, imgsz)

class ArPostProcessor(object):
    '''
    
    '''
    def __init__(self, n:int):
        self.n = n

    def __call__(self, prediction:np.ndarray) -> list:
        if len(prediction) == 2:
            prediction = np.squeeze(prediction, axis=0)
        classes = []
        for _ in range(self.n):
            idx = np.argmax(prediction)
            classes.append(idx)
            prediction[idx] -= 10000000000000000
        return classes

if __name__ == '__main__':
    test_engine_path = '../../model/posec3d_v1.engine'
    model = PoseC3dEngine(test_engine_path,17, 32, (224, 224))
    random_sequence = np.ndarray((1, 32, 17, 1, 56, 56), dtype=np.float32)
    out = model(random_sequence)
    
    ecopp = ArPostProcessor(10)
    ans = ecopp(out)
    print('done')