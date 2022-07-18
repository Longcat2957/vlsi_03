import cv2
import os
import subprocess
import numpy as np

class WriteVideo(object):
    def __init__(self, name:str, width:int, height:int, fps:int):
        self.name = name
        self.width, self.height = width, height
        self.fps = fps

    def __call__(self):
        return self._get_video_writer(self.name, self.width, self.height, self.fps)

    def _get_video_writer(self, name, width, height, fps=30):
        '''Get a VideoWriter object for saving output video.
        This function tries to use Jetson's hardware H.264 encoder (omxh264enc)
        if available, in which case the output video would be a MPEG-2 TS file.
        Otherwise, it uses cv2's built-in encoding mechanism and saves a MP4
        file.
        '''
        gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
        if 'omxh264dec' in gst_elements:
            filename = name + '.ts'
            gst_str = ('appsrc ! videoconvert ! omxh264enc ! mpegtsmux ! '
                   'filesink location=%s') % filename
            return cv2.VideoWriter(
                gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height)
            )
        else:
            filename = name + '.mp4'
            return cv2.VideoWriter(
                filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )



class LoadImage(object):
    
    def __init__(self):
        self.img_format = ['.jpeg', '.jpg', '.png']
        
    def _check_file(self, input_path):
        if os.path.exists(input_path):
            return True
        else:
            False
    
    def __call__(self, input_path):
        if self._check_file(input_path):
            _, format = os.path.splitext(input_path)
            if format in self.img_format:
                org_img = cv2.imread(input_path)
                shape = self._get_properties(org_img)
                shape['format'] = format
                return shape, org_img

            else:
                return False
        
        else:
            return None
        
    def _get_properties(self, img_obj):
        property = dict()
        H, W, C = img_obj.shape
        property['shape'] = (C, H, W)
        return property

class LoadVideo(object):
    '''
    Return property, cv2.VideoCapture(object)
    '''
    def __init__(self):
        self.video_format = ['.avi', '.mp4']
    
    def _check_file(self, input_path):
        if os.path.exists(input_path):
            return True
        else:
            False

    def __call__(self, input_path):
        if self._check_file(input_path):
            _, format = os.path.splitext(input_path)
            if format in self.video_format:
                cap = cv2.VideoCapture(input_path)
                property = self._get_properties(cap)
                property['format'] = format
                return property, cap
            else:
                return False
        else:
            return None
        
    def _get_properties(self, vid_obj):
        property = dict()
        C, H, W = 3, vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT), vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH)
        property['shape'] = (C, H, W)
        property['frame'] = vid_obj.get(cv2.CAP_PROP_FPS)
        property['codec'] = vid_obj.get(cv2.CAP_PROP_FOURCC)
        return property
        
        
if __name__ == '__main__':
    test_LoadVideo = LoadVideo()
    test_LoadVideo('./test2.mp4')
    test_LoadImage = LoadImage()
    shape, orig_img = test_LoadImage('../input/test1.jpeg')
    print('done')