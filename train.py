import json
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time


torch.manual_seed(66)

from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data):
        """
        X_data: 输入数据
        """
        self.X_data = X_data
    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx]).long()  # 转换为 Tensor
        return x

class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.nodeembedding = nn.Embedding(28436,300)
    def forward(self, x):
        res=self.nodeembedding(x)
        res1=res[:,0,:]
        res2=res[:,1,:]
        #边的权重如果定义了可以在这块乘上
        res_dot=res1*res2
        res_dot=res_dot.sum()
        return res_dot**2


if __name__=='__main__':
    node2id = np.load('node2id.npy',allow_pickle=True).item()
    nodeinit = np.load('node2init.npy',allow_pickle=True).item()
    nodevalue = np.load('nodevalue.npy',allow_pickle=True).item()
    embed=np.load('embed.npy')
    traindata=np.load('train.npy')
    train_tensor=torch.tensor(traindata)
    dataset = MyDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)#13003是质数。
    model=My_model()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(200):
        start=time.time()
        pre_count=0
        for batch_idx, (inputs) in enumerate(dataloader):
            optimizer.zero_grad()
            res=model(inputs)
            res.backward()
            # print("batch_idx:",batch_idx,"res:",res.detach().numpy())
            optimizer.step()
            pre_count+=res
            # break
        pre_count/=13003
        end=time.time()
        print("epoch:",epoch," loss:",pre_count.detach().numpy(),"time:",end-start,"s")
        