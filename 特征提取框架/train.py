#train and eval
import os,time,torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from dataload import FaceData
from torchvision.transforms import transforms
from tqdm import tqdm
from FeatureNet import  FeatureNet
from visualTrain import visual
BATCH_SIZE=24
EPOCH=1000

class Trainer:
    def __init__(self,net,save_path,dataset_path,log_path,is_Cuda=True):
        self.net=net
        self.save_path=save_path
        self.dataset_path=dataset_path
        self.log_path=log_path
        self.is_Cuda=is_Cuda
        if is_Cuda:
            self.net.cuda()

        #损失函数，NLLLoss arcface决定
        self.loss_fn=nn.NLLLoss()
        self.opt=opt.Adam(self.net.parameters())
        self.summary=SummaryWriter(self.log_path)
        if os.path.exists(self.save_path):
            files = os.listdir(self.save_path)
            # print(files)
            file = files[-1]
            print('load the {}'.format(file))
            # exit()
            file_path = os.path.join(self.save_path, file)
            self.net.load_state_dict(torch.load(file_path))

    def __call__(self):
        print("😬很有精神！")
        faceDataset=FaceData(self.dataset_path)
        dataloader=DataLoader(faceDataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)

        for epoch in tqdm(range(EPOCH),desc="😈好，模型很有精神!",ncols=70):
            start_time=time.time()
            loss_=torch.tensor([])
            # theta_=torch.tensor([])
            label_=torch.tensor([])
            feature_=torch.tensor([])
            for i,(data,label) in enumerate(dataloader):
                # print(data)
                # print(label)

                if self.is_Cuda:
                    data=data.cuda()
                    label=label.cuda()

                #计算损失，梯度反传，梯度更新
                feature,out=self.net(data)

                loss=self.loss_fn(out,label)
                loss_cpu=torch.tensor([loss.cpu().clone().detach()])
                loss_ = torch.cat((loss_,loss_cpu))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                feature_mean = torch.mean(feature, dim=1).cpu().detach()
                feature_=torch.cat((feature_,feature_mean))
                label_=torch.cat((label_,label.cpu().detach()))
                # print(feature_)
                # print(label_)

                #计算W与特征向量的弧度值
                # w=self.net.clasisfier.W
                # feature_nor = torch.nn.functional.normalize(feature, p=2, dim=1)
                # w_nor = torch.nn.functional.normalize(w, p=2, dim=0)
                # theta = torch.acos(torch.matmul(feature_nor, w_nor))
                # mean_theta = torch.mean(theta, dim=1)
                # mean_theta=mean_theta. pu().clone().detach()
                # mean_theta=torch.tensor(mean_theta)
                # theta_=torch.cat((theta_,mean_theta))

                #取特征额均值
            mean_loss = torch.mean(loss_)
            visual(feature_.data.cpu().detach(),label_.cpu().detach(),epoch,mean_loss)

            #存模型及相关信息
            # mean_theta_=torch.mean(theta_)

            end_time=time.time()
            print("第\033[1:31m【{}】\033[0m轮结束，耗时{}".format(epoch,end_time-start_time))
            # print('【loss】:%.4f,【theta】:%.4f'%(mean_loss,mean_theta_))
            # self.summary.add_scalars('loss',{'loss':mean_loss,'theta':mean_theta_},epoch)
            save_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel\model_epoch{}.pt".format(epoch)
            torch.save(self.net.state_dict(),save_path)
            print('模型存好了！')

net=FeatureNet(num_cls=4)
save_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\savedmodel"
log_path=r"C:\Users\liewei\Desktop\人脸识别\特征提取框架\logs"
datastet_path=r"C:\Users\liewei\Desktop\face_data"
trainer=Trainer(net,save_path,datastet_path,log_path)
trainer()





