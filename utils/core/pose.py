import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

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

