import torch
import torch.nn as nn


class IosLoss(nn.Module):
    def __init__(self, config):
        super(IosLoss, self).__init__()
        self.config = config

    def forward(self, true, false):
        # print("true",true)
        # print("false",false)
        # print("true sum",torch.sum(true, dim=(3,2,1)))
        # print("false sum",torch.sum(false, dim=(3,2,1)))
        # print("分母：",torch.sum(true, dim=(3,2,1))+torch.sum(false, dim=(3,2,1)))
        # print("mul ",torch.mul(true, false))
        # print("分子：",-2 * torch.sum(torch.mul(true, false), dim=(3,2,1)))

        # exit(0)


        denominator = (torch.sum(true, dim=(3,2,1)) + torch.sum(false, dim=(3,2,1)))#分母
        numerator = (-2 * torch.sum(torch.mul(true, false), dim=(3,2,1)))#分子
        iou =  numerator / denominator
        return torch.sum(iou) / self.config.batchs


        # return -2*torch.sum(torch.mul(true,false))/(torch.sum(true)+torch.sum(false))
        # iou = 2*torch.sum(torch.mul(true[0],false[0]))/torch.sum(true[0])+torch.sum(false[0])
        # i = 1
        # while i<self.config.batchs:
        #     iou+=2*torch.sum(torch.mul(true[i],false[i]))/torch.sum(true[i])+torch.sum(false[i])
        #     i+=1
        # return  -1*iou/self.config.batchs

