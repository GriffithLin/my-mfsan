import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from typing import Tuple, Optional, List, Dict

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class ADDneck(nn.Module):
# ADDneck(2048, 256)
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

# 定制: backbone，num_classes，finetune,  损失函数，avgpool
class MFSAN(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, finetune:bool,bottleneck_dim: Optional[int] = 2048, **kwargs):
        super(MFSAN, self).__init__()

        # self.bottleneck = nn.Sequential(
        #     nn.Linear(backbone.out_features, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU()
        # )

        self.sharedNet = backbone
        self.sonnet1 = ADDneck(backbone.out_features, 256)
        self.sonnet2 = ADDneck(backbone.out_features, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.finetune = finetune

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        mmd_loss = 0
        if self.training == True:
            if mark == 1:
                # 在公共空间提取特征
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                #先用子网络在子空间1提取目标域数据的特征
                #子空间 目标域
                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)

                #再同样在子空间1中，提取源域1数据特征. 计算目标域数据和源域1数据 在子空间1的mmd
                #子空间 源域 + mmd
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)

                #目标域数据 分类器1分类
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

                #目标域数据 分类器2分类
                #计算l1——loss
                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)

                #分类损失
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 2:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd.mmd(data_src, data_tgt_son2)

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)

                #Todo 注释语句可以用吗？
                #l1_loss = F.l1_loss(torch.nn.functional.softmax(data_tgt_son1, dim=1), torch.nn.functional.softmax(data_tgt_son2, dim=1))

                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            return pred1, pred2
    
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
    # """A parameter list which decides optimization hyper-parameters,
    #     such as the relative learning rate of each layer
    # """
        params = [
            {"params": self.sharedNet.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {'params': self.cls_fc_son1.parameters(), 'lr': 1.0 * base_lr},
            {'params': self.cls_fc_son2.parameters(), 'lr': 1.0 * base_lr},
            {'params': self.sonnet1.parameters(), 'lr': 1.0 * base_lr},
            {'params': self.sonnet2.parameters(), 'lr': 1.0 * base_lr},
        ]

        return params

