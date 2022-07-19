import numpy as np
import cv2
from .core.common import Cv2PreProcessor

# ROI_Extractor
# VideoSequence 객체

class CocoPoseObjects(object):
    '''
    CocoPoseObjects Class
    '''
    def __init__(self, line:bool):
        self.line = line
        self.coco_skeletons = [
            [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12], [5,6],[5,7],
            [6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]
        ]
    def __call__(self, pdets:np.ndarray):
        lines = self._get_lines(pdets)
        return pdets, lines

    def _get_lines(self, points:np.ndarray):
        if self.line:
            lines = []
            for l in self.coco_skeletons:
                p1, p2 = l[0], l[1]
                lines.append((points[p1], points[p2]))
            
            return lines
        else:
            return None

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
        try:
            n, _ = self.org_dets.shape
            class_array = self.org_dets[:, 5].flatten()
            valid = np.array([x in self.limitation for x in class_array])
            self.dets = self.org_dets[valid]
        except:
            pass


class VideoSequenceObjects(object):

    def __init__(self, frame=16, shape=(224, 224)):
        self.frame = frame
        self.shape = shape
        self.buffer = None
        self.tranformer = Cv2PreProcessor(shape)

    def __bool__(self):
        if isinstance(self.buffer, np.ndarray):
            return True
        else:
            return False

    def __len__(self):
        if isinstance(self.buffer, np.ndarray):
            T, C, H, W = self.buffer.shape
            return T
        else:
            return 0

    def clear(self):
        self.buffer = None

    def push(self, input):
        input = self._preproc(input)
        if self.buffer == None:
            self.buffer = input
        else:
            self.buffer = np.concatenate((self.buffer, input), axis=0)
    
    def pull(self):
        if isinstance(self.buffer, np.ndarray):
            length = len(self.buffer)
            
            if length > self.frame:
                stride = length // self.frame
                out = self.buffer[0:stride * (self.frame - 1) + 1:stride, :, :, :]
                self.clear()
                return out

            elif length == self.frame:
                out = self.buffer
                self.clear()
                return out
            
            elif length < self.frame:
                padding = self.frame - length
                surplus = np.zeros((padding, 3, 224, 224))
                out = np.concatenate((surplus, self.buffer), axis=0)
                self.clear()
                return out
        else:
            return None

    def _preproc(self, cv2_input:np.ndarray):
        tensor, _ = self.tranformer(cv2_input)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    

if __name__ == '__main__':

    sequence = VideoSequenceObjects()
    print(bool(sequence))
    print(len(sequence))
    
    random_frame = np.ndarray((1, 3, 224, 224))
    sequence.push(random_frame)

    print(bool(sequence))
    print(len(sequence))

    out = sequence.pull()
    print(out.shape)
    sequence.clear()

    print(bool(sequence))
    print(len(sequence))


    print('done')