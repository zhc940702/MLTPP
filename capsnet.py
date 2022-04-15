import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from capsule import CapsuleLayer


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.inputdim1 = 517
        self.inputdim2 = 5
        self.inputdim3 = 20
        self.outputdim = 512
        self.classnum = 5
        self.capsnum2 = 32
        self.capslen2 = 32
        self.capsnum3 = 32
        self.capslen3 = 32
        self.capsnum4 = 32
        self.capslen4 = 32

        self.feature1 = nn.Linear(self.inputdim1,self.outputdim)
        self.feature2 = nn.Linear(self.inputdim2,self.outputdim)
        self.feature3 = nn.Linear(self.inputdim3,self.outputdim)
        self.batchnorm = nn.BatchNorm1d(self.outputdim)
        self.primary_capsules = CapsuleLayer(num_capsules=7, num_route_nodes=-1, in_channels=self.outputdim, out_channels=32)
        self.digit_capsules = CapsuleLayer(num_capsules=self.capsnum2, num_route_nodes=7, in_channels=self.outputdim,
                                           out_channels=self.capslen2)
        self.digit_capsules2 = CapsuleLayer(num_capsules=self.capsnum3, num_route_nodes=self.capsnum2, in_channels=self.capslen2,
                                           out_channels=self.capslen3)
        # self.digit_capsules3 = CapsuleLayer(num_capsules=self.capsnum4, num_route_nodes=self.capsnum3,
        #                                     in_channels=self.capslen3,
        #                                     out_channels=self.capslen4)
        self.classify = nn.Linear(self.capsnum3*self.capslen3, self.classnum)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, f1,f2,f3,f4,f5,f6,f7,batchsize):
        x1 = self.feature1(f1)
        x2 = self.feature1(f2)
        x3 = self.feature1(f3)
        x4 = self.feature1(f4)
        x5 = self.feature2(f5)
        x6 = self.feature3(f6)
        x7 = self.feature3(f7)
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7),1)
        x = self.dropout(x)
        x = x.view([batchsize,-1,self.outputdim])
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = x.squeeze().transpose(0, 1)
        x = self.digit_capsules2(x)
        x = x.squeeze().transpose(0, 1)
        x = x.reshape([batchsize,-1])
        x = self.classify(x)
        x = torch.softmax(x,dim=1)
        return x


if __name__ == "__main__":
    model = CapsuleNet()
    print(model)
