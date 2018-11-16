import torch.nn as nn
import torch.nn.functional as F
import torch
class RepairNetBasedVgg16(nn.Module):
        def __init__(self,config):
            super(RepairNetBasedVgg16, self).__init__()

            #第一部分由4个卷积层、最大池化层、批正则化组成
            self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1)
            self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #self.bacht1 = nn.BatchNorm2d(num_features=32)

            #第二部分
            self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
            self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #self.batch2 = nn.BatchNorm2d(num_features=64)

            #第三部分
            self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
            self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #self.batch3 = nn.BatchNorm2d(num_features=128)

            #第四部分
            self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
            self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.pool4 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #self.batch4 = nn.BatchNorm2d(num_features=256)

            #第五部分
            self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
            self.pool5 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
            #self.bacht5 = nn.BatchNorm2d(num_features=512)

            #第六部分
            self.conv6_1 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=1,stride=1,padding=0)

        def forward(self, x):
            #第一部分前向传递
            x = F.relu(self.conv1_1(x))#1-1卷积输出
            x = F.relu(self.conv1_2(x))#1-2卷积输出
            x = self.pool1(x)
            #x = self.bacht1(x)

            #第二部分前向传递
            x = F.relu(self.conv2_1(x))#2-1卷积输出
            x = F.relu(self.conv2_2(x))  # 2-1卷积输出
            x = self.pool2(x)
            #x = self.batch2(x)

            #第三部分前向传递
            x = F.relu(self.conv3_1(x))#3-1卷积输出
            x = F.relu(self.conv3_2(x))  # 3-2卷积输出
            x = F.relu(self.conv3_3(x))  # 3-3卷积输出
            x = self.pool3(x)
            #x = self.batch3(x)

            #第四部分前向传递
            x = F.relu(self.conv4_1(x))#4-1卷积输出
            x = F.relu(self.conv4_2(x))  # 4-2卷积输出
            x = F.relu(self.conv4_3(x))  # 4-3卷积输出
            x = self.pool4(x)
            #x = self.batch4(x)

            #第五部分前向传递
            x = F.relu(self.conv5_1(x))#5-1卷积输出
            x = F.relu(self.conv5_2(x))  # 5-2卷积输出
            x = F.relu(self.conv5_3(x))  # 5-3卷积输出
            x = self.pool5(x)
            #x = self.bacht5(x)

            # x = F.relu()#6-1卷积输出
            x = torch.sigmoid(self.conv6_1(x))

            return x

class RepairNetBasedUnet(nn.Module):
    def __init__(self, config):
        super(RepairNetBasedUnet, self).__init__()
        # 第一部分由4个卷积层、最大池化层、批正则化组成
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bacht1 = nn.BatchNorm2d(num_features=32)

        # 第二部分
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch2 = nn.BatchNorm2d(num_features=64)

        # 第三部分
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch3 = nn.BatchNorm2d(num_features=128)

        # 第四部分
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch4 = nn.BatchNorm2d(num_features=256)

        # 第五部分
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bacht5 = nn.BatchNorm2d(num_features=512)

        # 第六部分
        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bacht6 = nn.BatchNorm2d(num_features=512)

        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 第一部分前向传递
        x = F.relu(self.conv1_1(x))  # 1-1卷积输出
        x = F.relu(self.conv1_2(x))  # 1-2卷积输出
        x = self.pool1(x)
        x = self.bacht1(x)

        # 第二部分前向传递
        x = F.relu(self.conv2_1(x))  # 2-1卷积输出
        x = self.pool2(x)
        x = self.batch2(x)

        # 第三部分前向传递
        x = F.relu(self.conv3_1(x))  # 3-1卷积输出
        x = self.pool3(x)
        x = self.batch3(x)

        # 第四部分前向传递
        x = F.relu(self.conv4_1(x))  # 4-1卷积输出
        x = self.pool4(x)
        x = self.batch4(x)

        # 第五部分前向传递
        x = F.relu(self.conv5_1(x))  # 5-1卷积输出
        x = self.pool5(x)
        x = self.bacht5(x)

        x = F.relu(self.conv6_1(x))  # 6-1卷积输出
        x = self.bacht6(x)
        x = torch.sigmoid(self.conv6_2(x))

        return x