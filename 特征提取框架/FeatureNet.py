#使用DenseNet121作为特征提取网络
import torch
import torch.nn as nn
from torchvision  import models
from 特征提取框架.arcloss import  LogArcFace
m=1.
s=10.

class FeatureNet(nn.Module):
    def __init__(self,num_cls):
        super(FeatureNet, self).__init__()
        self.feature_layer=models.densenet121(pretrained=True)
        self.feature_layer.classifier=nn.Linear(in_features=1024,out_features=512)#1024个特征
        self.clasisfier=LogArcFace(feature_nums=512,cls_nums=num_cls,margin=m,scale=s)

    def forward(self,x):
        feature=self.feature_layer(x)
        out=self.clasisfier(feature)
        return feature,out


# net=FeatureNet(300).cuda()
# data=torch.randn((1,3,200,200)).cuda()
# feat,out=net(data)
# w=net.clasisfier.W
# print(w.shape)
# print(feat.shape)
# print(out.shape)
# feat_nor=torch.nn.functional.normalize(feat,p=2,dim=1)
# w_nor=torch.nn.functional.normalize(w,p=2,dim=0)
# theta=torch.acos(torch.matmul(feat_nor,w_nor))
# print(theta)
# mean_theta=torch.mean(theta,dim=1)
# print(mean_theta.cpu().detach().item())