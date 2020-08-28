#定义dataloader
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import time
import matplotlib.pyplot as plt

class FaceData(Dataset):
    def __init__(self,path):
        start_time=time.time()
        super(FaceData, self).__init__()
        self.data_path=path
        dir = os.listdir(self.data_path)
        img_dir = []
        # print(dir)
        for i in dir:
            img_dir.append(os.path.join(self.data_path, str(i)))
        # print(img_dir)
        self.img_paths = []
        for i in img_dir:
            img_path = os.listdir(str(i))
            for j in img_path:
                self.img_paths.append(os.path.join(str(i), str(j)))
        end_time=time.time()
        print("数据集初始化完成!花了\033[1;31m%.3fs\033[0m"%(end_time-start_time))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path=self.img_paths[item]
        label=img_path.split("\\")[5]
        label=torch.tensor(int(label))
        img=Image.open(img_path)
        resizer=transforms.Resize((160,160))
        img=resizer(img)
        to_tensor=transforms.ToTensor()
        img_data=to_tensor(img)

        return img_data,label


# if __name__ == '__main__':
#     data_path=r"C:\Users\liewei\Desktop\face_data"
#     # dir=os.listdir(data_path)
#     # img_dir = []
#     # # print(dir)
#     # for i in dir:
#     #     img_dir.append(os.path.join(data_path,str(i)))
#     # # print(img_dir)
#     # img_paths=[]
#     # for i in img_dir:
#     #     img_path=os.listdir(str(i))
#     #     for j in img_path:
#     #         img_paths.append(os.path.join(str(i),str(j)))
#     # print(img_paths[10])
#     # data=img_paths[10]
#     # img=Image.open(img_paths[10])
#     # img.show()
#     # label=data.split("\\")[5]
#     # print(label)
#     data=FaceData(data_path)
#     b=data.__len__()
#     print(b)
#     c=data[10]
#     print(c)

#




