# pose predictor for NYU dataset

import cv2
from matplotlib.pyplot import imread
import torch
import numpy as np
from src.model import A2J_model
from src.anchor import A2JProcess

keypointsNumber = 14
cropWidth = 176
cropHeight = 176
downsample = 16
depth_thres = 150
depth_pixel_ratio = cropHeight / 2 / depth_thres

MEAN = np.load('/home/dejian/Project/A2J/data/nyu/nyu_mean.npy')
STD = np.load('/home/dejian/Project/A2J/data/nyu/nyu_std.npy')
model_dir = '/home/dejian/Project/A2J/model/nyu_mobilenetv2_8.71.pth'

class PosePredictor():
    def __init__(self):
        self.net = A2J_model(num_classes = keypointsNumber, backbone='mobilenet_v2')
        self.net.load_state_dict(torch.load(model_dir)) 
        self.net = self.net.cuda()
        self.net.eval()
        self.post_precess = A2JProcess(cropHeight, keypointsNumber, downsample)

    def _get_crop_img(self, img, center, lefttop, rightbottom):
        xmin = int(max(lefttop[0], 0))
        xmax = int(min(rightbottom[0], img.shape[1]-1))
        ymin = int(max(rightbottom[1], 0))
        ymax = int(min(lefttop[1], img.shape[0]-1))

        imCrop = img.copy()[int(ymin):int(ymax), int(xmin):int(xmax)]
        imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
        crop_img = imgResize.copy()

        imgResize = np.asarray(imgResize, dtype = 'float32') 

        imgResize[np.where(imgResize >= center[2] + depth_thres)] = center[2] 
        imgResize[np.where(imgResize <= center[2] - depth_thres)] = center[2] 

        imgResize = (imgResize - center[2]) * depth_pixel_ratio
        imgResize = (imgResize - MEAN) / STD


        imgResize = np.expand_dims(imgResize, 0)
        imgResize = np.expand_dims(imgResize, 0)
        return crop_img, imgResize  # (1,1,H,W)

    def _get_pred_keypoints(self, img, keypoints, center, lefttop, rightbottom):
        real_keypoints = keypoints.copy()
        real_keypoints[:, 0] = keypoints[:, 1]
        real_keypoints[:, 1] = keypoints[:, 0]

        xmin = max(lefttop[0], 0)
        xmax = min(rightbottom[0], img.shape[1]-1)
        ymin = max(rightbottom[1], 0)
        ymax = min(lefttop[1], img.shape[0]-1)

        real_keypoints[:,0] = real_keypoints[:,0]*(xmax-xmin)/cropWidth + xmin  # x
        real_keypoints[:,1] = real_keypoints[:,1]*(ymax-ymin)/cropHeight + ymin  # y
        real_keypoints[:,2] = real_keypoints[:,2] / depth_pixel_ratio + center[2]

        return real_keypoints


    def predict(self, img, center, lefttop, rightbottom):
        crop_img, img_data = self._get_crop_img(img, center, lefttop, rightbottom)
        img_data = torch.from_numpy(img_data).cuda()
        heads = self.net(img_data)
        keypoints = self.post_precess(heads)
        keypoints = keypoints.squeeze()
        keypoints = keypoints.cpu().data.numpy()
        keypoints = self._get_pred_keypoints(img, keypoints, center, lefttop, rightbottom)
        return crop_img, keypoints
