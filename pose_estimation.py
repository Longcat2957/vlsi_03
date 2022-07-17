import torch
import numpy as np
import cv2

from utils.io import LoadImage
from utils.core import Cv2PreProcessor
from utils.core import PoseHeatmapPostProcessor
from utils.obj import CocoPoseObjects
from utils.vis import CocoPoseVisualizer

if __name__ == '__main__':
    model = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True)
    model.eval()
    
    
    input_file_path = './input/test2.jpeg'
    image_loader = LoadImage()
    orig_property, orig_img = image_loader(input_file_path)
    print(orig_property)
    
    cv2pre = Cv2PreProcessor((256, 192))
    
    input_tensor, ratio = cv2pre(orig_img)
    input_tensor = input_tensor.reshape(1, 3, 256, 192)
    # print(input_tensor.shape)
    input_tensor = torch.tensor(input_tensor)
    output_tensor = model(input_tensor)
    output_tensor = output_tensor.detach().numpy()
    
    output_tensor = output_tensor.squeeze()
    # print(output_tensor.shape)
    
    posepost = PoseHeatmapPostProcessor()
    preds, _ = posepost(output_tensor, ratio)
    print(type(preds))
    print(f'preds, {preds}')

    cocoposeobj = CocoPoseObjects(True)

    pdets, lines = cocoposeobj(orig_property, preds)

    cocoposevis = CocoPoseVisualizer()
    orig_img = cocoposevis(orig_img, pdets, lines)


    cv2.imshow('test', orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()