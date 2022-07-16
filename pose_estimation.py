import torch
import numpy as np
import cv2

from utils.io import LoadImage
from utils.core import Cv2PreProcessor
from utils.core import PoseHeatmapPostProcessor

if __name__ == '__main__':
    model = torch.hub.load('yangsenius/TransPose:main', 'tph_a4_256x192', pretrained=True)
    model.eval()
    
    
    input_file_path = './input/human.jpeg'
    image_loader = LoadImage()
    shape, orig_img = image_loader(input_file_path)
    print(shape)
    
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
    preds, _ = posepost(output_tensor)
    preds *= 4
    
    preds[:, 0] *= (487/256)
    preds[:, 1] *= (386/192)
    preds = preds.astype(np.int16)
    # print(preds)
    
    def draw_point(orig_img, point:tuple):
        orig_img = cv2.line(orig_img, point, point, (0, 0, 255), 4)
        return orig_img
    
    for i in range(17):
        point = tuple(preds[i])
        orig_img = draw_point(orig_img, point)
        
    cv2.imshow('test', orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        