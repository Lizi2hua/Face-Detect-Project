import torch
import torch.nn as nn
import torch.nn.functional as F

class LogArcFace(nn.Module):
    def __init__(self,feature_nums,cls_nums,margin,scale):
        super(LogArcFace, self).__init__()
        self.W=nn.Parameter(torch.randn(feature_nums,cls_nums))
        self.m=margin
        self.s=scale
    def forward(self,features):
        #对features,W进行二范数归一化，使得归一化之后的模为1
        features_norm=F.normalize(features,p=2,dim=1)
        W_norm=F.normalize(self.W,p=2,dim=0)
        #求角度
        #feature_norm:[N,feature_nums]
        #W_nor:[feature_nums,cls_nums]
        #除以10减少梯度爆炸的概率，和W的初始化有关
        #由于W的参数是低维度的，所以很落在-1,1的周围的几率很大，而arccos在这两个值周围的梯度无穷大
        theta=torch.acos(torch.matmul(features_norm,W_norm)/10)
        # print(torch.matmul(features_norm,W_norm))
        # exit()
        #ArcFace->log
		# 	log\frac{e^{s(cos(\theta_{yi}+m))}}
		# 			{e^{s(cos(\theta_{yi}+m))}
		# 			+\sum_{j=1,j\neq yi}e^{scos\theta_{j}}}
        #分子
        numberator=torch.exp(self.s*torch.cos(theta+self.m))
        denominator=numberator+(torch.sum(torch.exp(self.s*torch.cos(theta)),dim=1,keepdim=True)-torch.exp(self.s*torch.cos(theta)))
        return torch.log(numberator/denominator)