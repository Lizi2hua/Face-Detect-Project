import cv2
import os
import json
from  PIL import Image
import numpy as np
import torch
from 侦测框架.core import detector
from 特征提取框架.extractor import Extractor
import time

thresh=0.70
data_path=r"C:\Users\liewei\Desktop\人脸识别\人脸检测框架\data"
dir=os.listdir(data_path)


#获取用户名，和数据的路径，每个user存有10个特征
users=[]
files=[]
for i in dir:
    user,_=os.path.splitext(i)
    file=os.path.join(data_path,i)
    files.append(file)
    users.append(user)
#读取全部的特征数据
buffer=[]
for j in files:
    with open(j,'r') as f:
        data=f.readlines()
        data=data[0]
        data=json.loads(data)
        buffer.append(data)



# a=np.array(buffer)
# print(a.shape)
#摄像头读取视频，得到特征
feature_model_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel"
detector=detector.Detector()
extractor=Extractor(feature_model_path)
video=cv2.VideoCapture(0)
min_size=100
while True:
    features=[]
    ret,frame=video.read()
    if ret:
        b,g,r=cv2.split(frame)
        img1 = cv2.merge([r, g, b])  # 转换为神经网络用的图片格式
        img = Image.fromarray(img1)  # 转换为网络用的格式
        # 图片输入网络，得到框
        boxes = detector.detect(img)
        # 如果检测到人
        if boxes.shape[0] != 0:
            for box in boxes:
                x1 = int(box[1])
                y1 = int(box[2])
                w = int(box[3] - x1)
                h = int(box[4] - y1)
                # 计算边长
                squre = min(w, h)
                # 提取的框的最小边长有个阈值
                if squre < min_size:
                    pass
                else:

                    cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + squre, int(y1 + squre)), (0, 255, 0),
                                  thickness=1)
                    img_crop = frame[y1 + 11:y1 + squre, x1 + 11:x1 + squre]

                    b, g, r = cv2.split(img_crop)
                    img_crop1 = cv2.merge([r, g, b])  # 转换为神经网络用的图片格式
                    img = Image.fromarray(img_crop1)  # 转换为网络用的格式
                    feature = extractor(img)  # [1,512]
                    feature = torch.squeeze(feature)  # [512]
                    feature = feature.cpu().detach().numpy()
                    feature = list(feature)
                    features.append(feature)  # list.append确保维度[n,512]，n表示一帧中有多少个框
                    #写这儿是为了实时性
                    #特征对比
                    thetas=[]
                    for id in range(len(files)):
                        data=buffer[id]
                        data=torch.tensor(data)
                        features=torch.tensor(features)
                        theta=torch.cosine_similarity(data,features,dim=1)
                        theta=torch.mean(theta)#取所有一个人所有特征与当前特征的余弦相似度的均值
                        thetas.append(theta)

                    thetas=torch.tensor(thetas)
                    index=torch.argmax(thetas,dim=0)
                    max_value=thetas[index]
                    if max_value>thresh:
                        target_user=users[index]
                        print(max_value)
                        print(thetas)
                        print(target_user)
                    else:
                        print(max_value)
                    # exit()


                    #将提取的特征做对比
    cv2.imshow("detecting",frame)
    cv2.waitKey(21)





