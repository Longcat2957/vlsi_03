import numpy as np
import cv2
from utils.core import Cv2PreProcessor

# ROI_Extractor
# VideoSequence 객체

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


class VideoSequenceObjects(object):

    def __init__(self, frame=16, shape=(224, 224)):
        self.frame = frame
        self.shape = shape
        
        self.transformer = Cv2PreProcessor(shape)
        self.buffer = None

    def __bool__(self):
        if isinstance(self.buffer, np.ndarray):
            return True
        else:
            return False

    def __len__(self):
        if self.__bool__():
            T, C, H, W = self.buffer.shape
            return T
        
        else:
            return 0

    def clear(self):
        self.buffer = None

    def push(self, input):
        if self.buffer == None:
            resized, _ = self.transformer(input)
            resized = np.reshape(resized, (1, 3, self.shape[0], self.shape[1]))
            self.buffer = resized
        else:
            input, _ = self.transformer(input)
            input = np.reshape(input, (1, 3, self.shape[0], self.shape[1]))
            self.buffer = np.concatenate(self.buffer, input, axis=0)
    
    def pull(self):
        copy = self.buffer
        return copy

if __name__ == '__main__':

    sequence = VideoSequenceObjects()
    print(bool(sequence))
    print(len(sequence))
    
    random_frame = np.ndarray((400, 600, 3))
    sequence.push(random_frame)

    print(bool(sequence))
    print(len(sequence))

    out = sequence.pull()
    print(out.shape)
    sequence.clear()

    print(bool(sequence))
    print(len(sequence))


    print('done')