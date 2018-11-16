from __future__ import print_function, division
import os,cv2
import torch
import numpy as np
from torch.utils.data import Dataset
class DataHandler(Dataset):
    """Repair Images Dataset"""

    def __init__(self,config,mode="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        if mode == "train":
            self.dataInfoList =self.config.dataInfoList[self.config.valDataCount:]
        elif mode == "val":
            self.dataInfoList = self.config.dataInfoList[0:self.config.valDataCount]


    def __len__(self):
        return len(self.dataInfoList)
    def __getitem__(self, idx):
        imgPath = os.path.join(self.dataInfoList[idx][1])#图片路径
        image = self.preProcess(imgPath)
        if self.config.mode=="train" or self.config.mode=="test":
            labelArray = self.dataInfoList[idx][2].transpose((2, 0, 1))#标记信息,from H x W x C to C X H X W
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(labelArray)}
        elif self.config.mode=="predict":
            pass
    def preProcess(self,imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = cv2.equalizeHist(img)  # 灰度图像直方图均衡化'
        img = cv2.resize(img, (self.config.shrinkImgWidth,self.config.shrinkImgHeight), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=2)  # 增加一个维度，
        img = img.astype('float32')  # 转换类型为float32
        img /= 255.

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        return img