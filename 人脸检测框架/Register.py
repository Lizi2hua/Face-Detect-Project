import cv2
from 侦测框架.core import detector
from  PIL import Image
from 特征提取框架.extractor import Extractor
import time
from tqdm import tqdm
import torch
import osbz
import numpy as np

#调试超参数
video_path=r"C:\Users\liewei\Desktop\tmr.mp4"
data_path=r"C:\Users\liewei\Desktop\人脸识别\人脸检测框架\data"
feature_model_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel"
# MTCNN_model_path=r"C:\Users\liewei\Desktop\人脸识别\侦测框架\saved_models"
class Register():
    def __init__(self):
        self.min_size=100
        self.detector=detector.Detector()
        self.extractor=Extractor(feature_model_path)
        self.frame_count=0
        self.video = cv2.VideoCapture(0)
    def __call__(self):
        #注册模式

        frame_skip=0
        features = []
        id=input("ID:")
        path = os.path.join(data_path, id)

        while self.frame_count in tqdm(range(10)):
            #提取时，跳10帧
            frame_skip += 1
            ret, frame = self.video.read()
            if ret:
                b, g, r = cv2.split(frame)
                img1 = cv2.merge([r, g, b])  # 转换为神经网络用的图片格式
                img = Image.fromarray(img1)  # 转换为网络用的格式
                # 图片输入网络，得到框
                boxes = self.detector.detect(img)
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
                        if squre < self.min_size:
                            pass
                        else:
                            if frame_skip % 10 == 0:
                                cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + squre, int(y1 + squre)), (0, 255, 0),
                                              thickness=1)
                                img_crop = frame[y1 + 11:y1 + squre, x1 + 11:x1 + squre]

                                b, g, r = cv2.split(img_crop)
                                img_crop1 = cv2.merge([r, g, b])  # 转换为神经网络用的图片格式
                                img = Image.fromarray(img_crop1)  # 转换为网络用的格式
                                feature =self.extractor (img)#[1,512]
                                feature=torch.squeeze(feature)#[512]
                                feature=feature.cpu().detach().numpy()
                                feature=list(feature)
                                features.append(feature)#[12,512]

                                self.frame_count += 1
                            else:
                                pass
            cv2.imshow('register ID', frame)
            cv2.waitKey(21)
        #写入txt
        cv2.destroyAllWindows()

        print("写入中...")
        txt = open('{}.txt'.format(path), 'w')
        txt.write(str(features))
        txt.close()
        print("写入成功")
        print(len(features))

v=Register()
v()





