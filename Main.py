from Config import Config
from DataSetTools import DataHandler
from IouLoss import IosLoss
import argparse
from Net import RepairNetBasedUnet,RepairNetBasedVgg16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Some Variable parameters.')
    parser.add_argument("-e","--epochs",dest="epochs",help="enter -e epochs(optional,default=2000)",default=2000)
    parser.add_argument("-m","--mode",dest="mode",help="train mode enter -m train,test mode enter -m test predict mode enter -m predict",default="train")
    parser.add_argument("-d","--testDictinary ",dest="testDic",help="specify a test data dictionary",default=None)
    parser.add_argument("-v","--valDataCount",dest="valDataCount",help="valuate data count",default=200)
    args = parser.parse_args()

    config = Config()

    if args.mode == "train":
        config.epochs = args.epochs
        config.mode = args.mode
        config.valDataCount = args.valDataCount

        model = RepairNetBasedUnet(config)
        #model = RepairNetBasedVgg16(config)
        trainData = DataHandler(config,mode="train")
        trainDataLoader = DataLoader(trainData, batch_size=config.batchs,shuffle=True, num_workers=4)

        valData = DataHandler(config,mode="val")
        valDataLoader = DataLoader(valData, batch_size=config.batchs,shuffle=True, num_workers=4)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        #定义损失函数和优化函数
        criterion = IosLoss(config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        #开始训练
        for epoch in range(config.epochs):  # loop over the dataset multiple times
            runningLoss = 0.0
            for i, data in enumerate(trainDataLoader, 0):
                # get the inputs
                inputs = data["image"]
                labels = data["label"]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs.to(device))

                loss = criterion(labels.to(device=device,dtype=torch.float), outputs)
                loss.backward()
                optimizer.step()

                # print statistics
                runningLoss += loss.item()
                #print("running loss:",loss.item())
                if i % 200 == 199:  # print every 200 mini-batches
                    print('train epoch %d, mini-batches %5d loss: %.3f' % (epoch + 1, i + 1, runningLoss / 200))
                    runningLoss = 0.0

            #进行验证
            # 把模型设为验证模式
            model.eval()
            runningLoss = 0.0
            valTotalBatchNum = 0
            for i, data in enumerate(valDataLoader, 0):
                # get the inputs
                inputs = data["image"]
                labels = data["label"]

                outputs = model(inputs.to(device))
                loss = -1*criterion(labels.to(device=device,dtype=torch.float),outputs)
                runningLoss += loss.item()
                valTotalBatchNum = i
            print('train epoch:%d, val batch :%5d,val loss: %.3f' % (epoch + 1, valTotalBatchNum + 1, runningLoss / (valTotalBatchNum+1)))
            runningLoss = 0.0

            # 把模型恢复为训练模式
            model.train()
        print('Finished Training')
