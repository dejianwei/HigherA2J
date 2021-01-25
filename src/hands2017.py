import cv2
import torch
import random
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
import time
from PIL import Image
import model as model
import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
from ptflops import get_model_complexity_info

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx = 475.065948
fy = 475.065857
u0 = 315.944855
v0 = 245.287079

validIndex_train = np.load('./data/hands2017/validIndex.npy')
TrainImgFrames = len(validIndex_train)  # 957032
TestImgFrames = 295510
validIndex_test = np.arange(TestImgFrames)
keypointsNumber = 21
cropWidth = 176
cropHeight = 176
batch_size = 32
xy_thres = 100
depth_thres = 150
depth_pixel_ratio = cropHeight / 2 / depth_thres
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 17
downsample = 16

RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180 
RandScale = (1.0, 0.5)
randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = './result/HANDS2017'

try:
    os.makedirs(save_dir)
except OSError:
    pass

trainingImageDir = '/home/dejian/Dataset/hands2017/images/'
train_center_file = './data/hands2017/hands2017_center_train.mat'
train_keypoint_file = './data/hands2017/hands2017_keypointsUVD_train.mat'

testingImageDir = '/home/dejian/Dataset/hands2017/frame/images/'
test_center_file = './data/hands2017/hands2017_center_test.mat'
test_keypoint_file = './data/hands2017/hands2017_keypointsUVD_test.mat'

MEAN = np.load('./data/hands2017/hands2017_mean.npy')
STD = np.load('./data/hands2017/hands2017_std.npy')
model_dir = './model/hands2017_resnet50.pth'
result_file = 'result_HANDS2017.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x

# train
keypointsUVD_train = scio.loadmat(train_keypoint_file)['keypoints3D'].astype(np.float32)   

center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-xy_thres
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+xy_thres
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-xy_thres

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)

# test
keypointsUVD_test = np.zeros((TestImgFrames,keypointsNumber,3),dtype=np.float32)

center_test = scio.loadmat(test_center_file)['centre_pixel'].astype(np.float)
centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)

def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]   
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=100, depth_thres=150, augment=False):
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 

    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
 
    new_Xmin = max(lefttop_pixel[index,0,0], 0)
    new_Ymin = max(rightbottom_pixel[index,0,1], 0)  
    new_Xmax = min(rightbottom_pixel[index,0,0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1], img.shape[0] - 1)

    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32') 

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2] 
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std
    
    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) 
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) 

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale
    
    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1] 
    # labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])   # Z  
    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])*depth_pixel_ratio
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, validIndex, augment=False):

        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres
        self.validIndex = validIndex
        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])

    def __getitem__(self, index):
        index = self.validIndex[index]
        depth = Image.open(self.ImgDir + 'image_D%.8d'%(index+1) + '.png') 
        depth = np.array(depth)
        
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres, self.augment)
        
        if self.augment:
            data = self.randomErase(data)
            
        return data, label
    
    def __len__(self):
        return len(self.center)

train_image_datasets = my_dataloader(trainingImageDir, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, validIndex_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size, shuffle = True, num_workers = 8)
      
test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, validIndex_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size, shuffle = False, num_workers = 8)

def train():
    net = model.A2J_model(num_classes=keypointsNumber, backbone='resnet50')
    net = net.cuda()
    
    post_precess = anchor.A2JProcess(cropHeight, keypointsNumber, downsample)
    criterion = anchor.A2JLoss(cropHeight, keypointsNumber, downsample)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')
    macs, params = get_model_complexity_info(net, (1,cropHeight,cropWidth),as_strings=True,print_per_layer_stat=False)
    logging.info('macs=%s, params=%s'%(macs, params))

    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()
    
        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):

            torch.cuda.synchronize() 

            img, label = img.cuda(), label.cuda()     
            
            heads  = net(img)  
            optimizer.zero_grad()  
            
            Cls_loss, Reg_loss = criterion(heads, label)

            loss = Cls_loss + Reg_loss
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            
            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)

            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))

        saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch)
        torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, scheduler.get_lr()[0]))

     
def test():   
    net = model.A2J_model(num_classes=keypointsNumber, backbone='resnet50')
    net.load_state_dict(torch.load(model_dir)) 
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.A2JProcess(cropHeight, keypointsNumber, downsample)
    
    output = torch.FloatTensor()
    
    torch.cuda.synchronize() 
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):    
        with torch.no_grad():
            img, label = img.cuda(), label.cuda()             
            heads = net(img)  
            pred_keypoints = post_precess(heads)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
        
    torch.cuda.synchronize()       
    result = output.cpu().data.numpy()
    writeTxt(result, center_test)
   

def errorCompute(result, target, center):
    assert np.shape(result)==np.shape(target), "source has different shape with target"

    resultUVD_ = result.copy()
    target_ = target.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] / depth_pixel_ratio + center[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    resultXYD = pixel2world(resultUVD.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - resultXYD) ** 2, axis=2))

    return np.mean(errors)

def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 320*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 240*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] / depth_pixel_ratio + center[i][0][2]

    resultXYD = pixel2world(resultUVD.copy(), fx, fy, u0, v0)

    resultReshape = resultXYD.reshape(len(resultXYD), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:     
        for i in tqdm(range(len(resultReshape))):
            f.write('frame/images/' + 'image_D%.8d'%(i+1) + '.png' + '\t')
            for j in range(keypointsNumber*3):
                f.write(str(resultReshape[i, j])+'\t')
            f.write('\n') 

    f.close()


if __name__ == '__main__':
    # train()
    test()
