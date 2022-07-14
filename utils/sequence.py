import numpy as np
import cv2
from core import YoloPreProcessor

# ROI_Extractor
# VideoSequence 객체

class VideoSequence(object):

    def __init__(self, frame=16, shape=(224, 224)):
        self.frame = frame
        self.shape = shape
        
        self.transformer = YoloPreProcessor(shape)
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
        
        else:   # if self.buffer = None
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

    sequence = VideoSequence()
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