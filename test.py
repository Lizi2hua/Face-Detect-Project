import torch
import torch.nn as nn
from tqdm import tqdm
import time
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
# a=np.random.randn(28,28,3)
# img=Image.fromarray(a)
# to=transforms.ToTensor()
# a=to(img)
# print(a)
# sm=nn.Softmax(dim=1)
# a=sm(a)
# target=torch.tensor([1,1,1])
# a=torch.log(a)
# loss=nn.NLLLoss()
# c=loss(a,target)

# save_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel"
# if os.path.exists(save_path):
#    files=os.listdir(save_path)
#    print(files)
#    file=files[-1]
#    print(file)
#    file_path=os.path.join(save_path,file)
#    print(file_path)
# b=torch.tensor([[1,2],[2,1]])
# c=torch.tensor([[3,4],[4,3]])
# d=torch.stack((b,c),dim=1)
# print(d)
# c=torch.randn((3,160,160))
# cont=0
# while cont in tqdm(range(10)):
#     a=torch.randn(100)
#     b=torch.randn(100)
#     for i in range(100):
#         if a[i]>b[i]:
#             cont+=1
#             time.sleep(0.1)
# c_path=os.path.abspath(__file__)
# father_path = os.path.abspath(os.path.dirname(c_path) + os.path.sep + ".")
# print(father_path)
#
# a='s'
# b=[[1,2],[3,1]]
# c={a:b}
# print(c)
a=[[[1,2.],[2,3]],[[3,4],[5,6]],[[7,8],[9,8]]]
b=[[[1,2],[2,3]],[[3,4],[5,6]],[[7,8],[9,8.]]]
a=torch.tensor(a)
b=torch.tensor(b)
a=torch.nn.functional.normalize(a,p=2)
b=torch.nn.functional.normalize(b,p=2)
# b=b.permute(1,0,2)
a_=a[1][1]
b_=b[1][1]
print(a_)
print(b_.t())

theta=torch.cosine_similarity(a,b,dim=1)
print(theta)
c=[]
d=[1,2,3]
c.append(d)
print(c)

c=torch.tensor([0.1,0.4])
b=torch.max(c,dim=1)
print(c)