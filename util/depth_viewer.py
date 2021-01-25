
import numpy as np
import cv2
import random

img_path = '/home/dejian/Dataset/hands2017/frame/images/image_D00000001.png'
image = cv2.imread(img_path, -1)
depth = np.asarray(image, dtype=np.float32)

depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RealSense', depth_colormap)
cv2.waitKey(0)