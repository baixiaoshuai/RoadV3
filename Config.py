import time,os
import numpy as np
import torch,random,cv2
class Config:
    def __init__(self):

        # 训练和测试都需要的配置参数
        self.dataRootPath = "/home/public/rmt/road/data"  # 数据所在的根目录
        self.date = self.getDate()  # 获取当前时间
        self.dataInfoList = [] #获取数据的数据所有信息


        # 训练时需要的配置参数
        self.epochs = 2000  # 训练的epoch
        self.batchs = torch.cuda.device_count()*4  # 训练时设置的batch=gpu数量*4
        self.weightSavePath = '/home/public/rmt/road/result/bai/model/Model-' + self.date + '.tar'  # 设置权重保存的路径
        self.goOnTrain = False  # 是否接着训练
        self.valDataCount = '200'  # 设定作为测试数据的数量，'all'表示全部

        # 测试时使用的参数
        self.testWeightPath = ""  # 测试模型参数存放位置
        self.threshold = 0.5  # 阈值大小
        self.testDic = None #当测试数据都在一个路径的情况下使用

        self.mode = "train"  # 程序运行模式："train" or "test"
        self.dataRelativePathList = [] #程序数据的相对数据路径
        self.rawImageHeight = 2100  # 原始图片行数和列数（像素），实际像素小于该值，空白处补0
        self.rawImageWidth = 3200
        self.rawBlockSize = 100  # 原始块大小，长宽皆为100像素
        self.shrinkImgHeight = 672# 训练时压缩后图像像素为672*1024
        self.shrinkImgWidth = 1024
        self.shrinkBlockSize = 32# 压缩后块大小为长宽32像素

        self.blocksOfHeight = 21# 块的高和宽，总共21*32块
        self.blocksOfWidth = 32
        self.totalBlocksCount = (self.blocksOfHeight) * (self.blocksOfWidth)

        #根据训练模式，获取所需数据
        if self.mode == "train":
            self.addTrainData()
        else:
            self.addTestData()

        self.getDataPathList()
        print("总共有数据：%d" % (len(self.dataInfoList)))

        random.shuffle(self.dataInfoList)#随机打乱数据

    def getDate(self):
        '''
        获取当前日期信息
        :return: 返回日期字符串
        '''
        # if self.mode == "train":
        #     date = str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
        # else:
        date = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        return date

    def parseFileindex(self, path,mode="img"):
        ret = {}
        with open(os.path.join(path,"fileindex.txt"), "r") as f:  # 获取一个列表
            contents = f.readlines()  # 区别read(),readline()和readlines()
            for line in contents:
                lineMeta = line.split("->")
                relativePath = lineMeta[2].replace('\\', '/').strip("\n")[1:]
                imgName = lineMeta[1].strip()
                #print(os.path.join(path,relativePath,imgName))
                #exit(0)
                if mode=="img" and os.path.exists(os.path.join(path,relativePath,imgName)):#判断文件是否存在
                    ret[imgName] = os.path.join(path,relativePath,imgName)
                elif mode=="label" and os.path.exists(os.path.join(path,relativePath,imgName.replace(".jpg",".txt"))):
                    ret[imgName] = os.path.join(path,relativePath,imgName.replace(".jpg",".txt"))
        return ret
    def getDataPathList(self):
        num = 0
        if os.path.exists("index.txt"):
            total = 0
            with open("index.txt","r",encoding="utf-8") as f:
                for line in f.readlines():
                    meta = line.strip().split(",")
                    total+=1
                    if len(meta)==3:
                        if self.mode == "train" or self.mode == "test":  # 当模式是训练和测试时，有正确标记数组
                            labelArray = self.getLabelArray(meta[2])
                            self.dataInfoList.append((meta[0], meta[1], labelArray))
                        elif self.mode == "predict":  # 当模式是predict时，无正确标记数组
                            pass

                    if(total==5200):
                        return
        else:
            with open("index.txt","a+",encoding="utf-8") as f:
                for imgRelativePath,labelRelativePath in self.dataRelativePathList:
                    # 获取图片和标记文件的fileIndex目录
                    imgFileIndexPath = os.path.join(self.dataRootPath, imgRelativePath)
                    if self.mode=="train" or self.mode=="test":#当模式是训练和测试时，有正确标记数组
                        labelFileIndexPath = os.path.join(self.dataRootPath, labelRelativePath)

                        imgTempInfoDic = self.parseFileindex(imgFileIndexPath)
                        labelTempInfoDic = self.parseFileindex(labelFileIndexPath,mode="label")
                        print("road:%s,image:%d,label:%d" % (imgRelativePath,len(imgTempInfoDic),len(labelTempInfoDic)))
                        keys = imgTempInfoDic.keys() & labelTempInfoDic.keys() #获取两个字典共有的key
                        print("total key:",len(keys))
                        for key in keys:
                            labelArray = self.getLabelArray(labelTempInfoDic[key])
                            if np.sum(labelArray)>0:
                                f.write(key+","+imgTempInfoDic[key]+","+labelTempInfoDic[key]+"\n")
                                num += 1
                                self.dataInfoList.append((key,imgTempInfoDic[key],labelArray))
                                if num%1000==0:
                                    print("valid file %d" % (num))
                    elif self.mode=="predict":#当模式是predict时，无正确标记数组
                        pass

    def getLabelArray(self,labelPath):
        labelArray = np.zeros((self.blocksOfHeight,self.blocksOfWidth, 1), np.uint8)  # 生成一个标记数组
        try:
            with open(labelPath, "r") as l:
                lines = l.readlines()
                repairArray = np.zeros((self.blocksOfHeight, self.blocksOfWidth), np.uint8)
                maxCol = 0  # 记录数组最大的列
                maxRow = 0  # 记录最大的行
                for line in lines:
                    if line.find("Bad_RepairBlockPos") != -1:
                        index = line.strip("\n").replace("Bad_RepairBlockPos:", "").split(",")
                        index = tuple(int(x) for x in index)
                        if index[0] <= self.blocksOfHeight and index[1] <= self.blocksOfWidth:
                            repairArray[index[0] - 1, index[1] - 1] = 1
                            if index[0] > maxRow: maxRow = index[0]
                            if index[1] > maxCol: maxCol = index[1]
                # 去除块修只保留条修
                for iy in range(self.blocksOfHeight):
                    for ix in range(self.blocksOfWidth):
                        if repairArray[iy, ix] == 1:
                            if (iy + 3) <= self.blocksOfHeight:
                                if (ix + 3) <= self.blocksOfWidth and np.sum(repairArray[iy:iy + 3, ix:ix + 3]) == 9:
                                    continue
                                if (ix + 2) <= self.blocksOfWidth and (ix - 1) >= 0 and np.sum(repairArray[iy:iy + 3, ix - 1:ix + 2]) == 9:
                                    continue
                                if (ix + 1) <= self.blocksOfWidth and (ix - 2) >= 0 and np.sum(repairArray[iy:iy + 3, ix - 2:ix + 1]) == 9:
                                    continue
                            if (iy + 2) <= self.blocksOfHeight:
                                if (ix + 3) <= self.blocksOfWidth and np.sum(repairArray[iy - 1:iy + 2, ix:ix + 3]) == 9:
                                    continue
                                if (ix + 2) <= self.blocksOfWidth and (ix - 1) >= 0 and np.sum(repairArray[iy - 1:iy + 2, ix - 1:ix + 2]) == 9:
                                    continue
                                if (ix + 1) <= self.blocksOfWidth and (ix - 2) >= 0 and np.sum(repairArray[iy - 1:iy + 2, ix - 2:ix + 1]) == 9:
                                    continue
                            if (iy + 1) <= self.blocksOfHeight:
                                if (ix + 3) <= self.blocksOfWidth and np.sum(repairArray[iy - 2:iy + 1, ix:ix + 3]) == 9:
                                    continue
                                if (ix + 2) <= self.blocksOfWidth and (ix - 1) >= 0 and np.sum(repairArray[iy - 2:iy + 1, ix - 1:ix + 2]) == 9:
                                    continue
                                if (ix + 1) <= self.blocksOfWidth and (ix - 2) >= 0 and np.sum(repairArray[iy - 2:iy + 1, ix - 2:ix + 1]) == 9:
                                    continue
                            labelArray[iy, ix, 0] = 1  # 记录条状修补存在的位置
                # 去掉在图像四周的修补
                # labelArray[0,:,0]=0
                # labelArray[:,0,0]=0
                # labelArray[:,maxCol-1,0]=0
                # labelArray[maxRow-1,:,0]=0
                return labelArray
        except UnicodeDecodeError as e:
            return labelArray
            #pass

    def addTrainData(self):

        #trainRoadList = [("20170426/【精修样本】-20170405/江苏/Images/S245A", "20170426/【精修样本】-20170405/江苏/识别结果-合并/S245AM")]
        trainRoadList = [("20170426/【精修样本】-20170405/内蒙古/images/G72HB", "20170426/【精修样本】-20170405/内蒙古/识别结果-合并/G72HBM4"),
            ("20170426/【精修样本】-20170405/江苏/Images/S245A", "20170426/【精修样本】-20170405/江苏/识别结果-合并/S245AM"),
            ("20170426/【精修样本】-20170407/吉林/Images/GA12A", "20170426/【精修样本】-20170407/吉林/识别结果-合并/GA12A2M3"),
            ("20170426/【精修样本】-20170407/福建/Images/G205A", "20170426/【精修样本】-20170407/福建/识别结果-合并/G205A2M1"),
            ("20170426/【精修样本】-20170407/福建/Images/G205A", "20170426/【精修样本】-20170407/福建/识别结果-合并/G205A3M1"),
            ("20170426/【精修样本】-20170414/湖北浅色-20170414/Images/G316A","20170426/【精修样本】-20170414/湖北浅色-20170414/识别结果-合并/G316A6M2"),
            ("20170426/【精修样本】-20170414/河北浅色-20170414/Images/G205A", "20170426/【精修样本】-20170414/河北浅色-20170414/识别结果-合并/G205A1M1"),
            ("20170426/【精修样本】-20170426/甘肃/Images/G310A", "20170426/【精修样本】-20170426/甘肃/识别结果-合并/G310A2M3"),
            ("20170426/【精修样本】-20170426/甘肃/Images/G310A", "20170426/【精修样本】-20170426/甘肃/识别结果-合并/G310A3M4"),
            ("20170426/【精修样本】-20170426/甘肃/Images/G310A", "20170426/【精修样本】-20170426/甘肃/识别结果-合并/G310A4M4"),
            ("20170426/【精修样本】-20170426/四川/images/20170104/GA05A", "20170426/【精修样本】-20170426/四川/识别结果-修补/GA05A1M1"),
            ("20170426/【精修样本】-20170426/四川/images/20170107/GA65A", "20170426/【精修样本】-20170426/四川/识别结果-修补/GA65AM4")]
        for data in trainRoadList:
            self.dataRelativePathList.append(data)

    def addTestData(self):
        testRoadList = [("20170426/【精修样本】-20170405/内蒙古/images/GA78A", "20170426/【精修样本】-20170405/内蒙古/识别结果-合并/GA78A2M1"),
            ("20170426/【精修样本】-20170405/江苏/Images/S328A", "20170426/【精修样本】-20170405/江苏/识别结果-合并/S328AM3"),
            ("20170426/【精修样本】-20170407/吉林/Images/G201A", "20170426/【精修样本】-20170407/吉林/识别结果-合并/G201A5M"),
            ("20170426/【精修样本】-20170414/湖北浅色-20170414/Images/G316A","20170426/【精修样本】-20170414/湖北浅色-20170414/识别结果-合并/G316A7M3"),
            ("20170426/【精修样本】-20170426/四川/images/20170104/GA05A", "20170426/【精修样本】-20170426/四川/识别结果-修补/GA05A2M2")]
        for data in testRoadList:
            self.dataRelativePathList.append(data)
