import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path as osp
from models.base_model import BaseModel
from models.nn.ir_se_model import IR_50
from models.nn.res_se_34l_model import ResNetSE34
from models.nn.module import NormalizeConv, MultiLayerPerceptron, Conv1dModule, LinearModule


class SEResNet50IR(BaseModel):
    def __init__(self, args):
        super(SEResNet50IR, self).__init__(args)
        output_channel = self.args.output_channel
        self.model = IR_50([112, 112], pretrained='pretrained_models/backbone_ir50_ms1m_epoch120.pth')
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_channel, bias=False),           ## arg.output_channel = 128     The dimention of output of face is 128
            nn.Dropout(0.5)
        )
        self.cls = Classifier(output_channel, self.args.num_classes, self.args.vote)     ## cls(input=128, output=924)

    def forward(self, x, y=None):
        x = self.model(x)
        x = self.fc(x)
        return x

class ThinResNet34(BaseModel):
    def __init__(self, args):
        super(ThinResNet34, self).__init__(args)
        output_channel = self.args.output_channel
        self.model = ResNetSE34(pretrained='pretrained_models/baseline_lite_ap.model')
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_channel, bias=False),          ## also the dimention is 128
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, nin, nout, vote=False):
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nout, nin))        ## (924,128)
        nn.init.xavier_uniform_(self.weight)

    def dist(self, a, b):
        dist = (a * b).sum(-1)
        return dist

    def arc_margin(self, x, y, margin):              ## x.shape=(64, 128) y.shape=(64, 1)
        dist = self.dist(self.weight.unsqueeze(0), x.unsqueeze(1)) # N x M      (1,924,128)  (64, 1, 128)    dist.shape=(64, 924, 128).sum(-1).shape=(64, 924)
        one_hot = torch.zeros(dist.size()).to(x.device)            ## one_hot.shape=(64, 924)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)               ## 将y（shape=(64,1)）转为one-hot编码(64, 924)
        if margin is None:
            logit = (one_hot * (dist)) + ((1.0 - one_hot) * dist)      ## 不知为何 似乎没有必要    shape = (64, 924)
        else:
            logit = (one_hot * (dist - margin.unsqueeze(1))) + ((1.0 - one_hot) * dist)
        return logit

    def cross_logit(self, x, v):
        dist = self.dist(F.normalize(x).unsqueeze(0), v.unsqueeze(1))               ## self.dist(shape=(1,64,128),shape=(64,1,128)).shape = (64, 64)
        one_hot = torch.zeros(dist.size()).to(x.device)                             ## one_hot.shape=(64, 64)
        one_hot.scatter_(1, torch.arange(len(x)).view(-1, 1).long().to(x.device), 1)     ## 比较的是两模态输入的相似度logit，所以index i=j时，one-hot元素为1

        pos = (one_hot * dist).sum(-1, keepdim=True)        ## (64,1)
        logit = (1.0 - one_hot) * (dist - pos)              ## ???
        loss = torch.log(1 + torch.exp(logit).sum(-1) + 3.4)
        return loss

    def forward(self, x, y, margin=None):
        logit = self.arc_margin(x, y, margin)
        return logit
