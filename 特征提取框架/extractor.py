import torch
import torch.nn as nn
from 特征提取框架.FeatureNet import   FeatureNet
import os
from torchvision.transforms import transforms
import time


class Extractor():
    def __init__(self,model_path):
        self.net=FeatureNet(num_cls=4)
        self.net=self.net.cuda()
        self.model_path=model_path
        self.to_tensor = transforms.ToTensor()
        #只取最后一个模型文件):
        if os.path.exists(self.model_path):
            files = os.listdir(self.model_path)
            file = files[-1]
            file_path = os.path.join(self.model_path, file)
            self.net.load_state_dict(torch.load(file_path))
            self.net.eval()
        else:
            print("无效的文件路径")
    def __call__(self,img_data):
        img_data=self.to_tensor(img_data)#[CHW]
        #这行代码不需要，测试用
        img_data=torch.unsqueeze(img_data,dim=0)#[1,CHW]

        img_data=img_data.cuda()
        # print(img_data.shape)
        # exit()
        start_time=time.time()
        feature,_=self.net(img_data)
        end_time=time.time()
        print("特征提取完成，花了%.4fs"%(end_time-start_time))
        return feature

# from PIL import Image
# img_path = r"C:\Users\liewei\Desktop\face_data\1\1.jpg"
# model_path = r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel"
# extract=Extractor(model_path)
# img=Image.open(img_path)
# print(img)
# # exit()
# feat=extract(img)
# print(feat.shape)



