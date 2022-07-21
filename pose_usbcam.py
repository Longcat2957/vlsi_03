import cv2
from utils.io import LoadImage
from utils.core.common import Cv2PreProcessor
from utils.core.pose import TransPoseEngine, PoseHeatmapPostProcessor
from utils.obj import CocoPoseObjects
from utils.vis import CocoPoseVisualizer
from tools.benchmark import timer


class PoseInferenceBlock(object):
    def __init__(self, engine_path, joints, imgsz):
        self.preprocessor = Cv2PreProcessor(imgsz)
        self.tensorrt_engine = TransPoseEngine(engine_path, joints, imgsz)
        self.postprocessor = PoseHeatmapPostProcessor()

    def __call__(self, input):
        return self.infer(input)

    @timer
    def infer(self, input):
        x, r = self.preprocessor(input)
        x = self.tensorrt_engine(x)
        x = self.postprocessor(x, r)
        return x

if __name__ == '__main__':

    engine_path = './model/transpose-h.engine'
    inference_block = PoseInferenceBlock(engine_path, 17, (256, 192))
    video_idx = 0

    cam = cv2.VideoCapture('input/yoyo.mp4')

    # WIDTH = 640
    # HEIGHT = 480
    # FRAMERATE = 30

    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # cam.set(cv2.CAP_PROP_FPS, FRAMERATE)

    while cv2.waitKey(1) < 1:
        grabbed, orig_img = cam.read()
        if not grabbed:
            exit()


        preds, _ = inference_block(orig_img)    #discard maxvals
        pose_obj = CocoPoseObjects(True)
        pdets, lines = pose_obj(preds)


        pose_vis = CocoPoseVisualizer()
        final = pose_vis(orig_img, pdets, lines)

        cv2.imshow('image', final)

    cv2.destroyAllWindows()

    print('done')

