import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score

import numpy as np
import pandas as pd
import time
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import matplotlib.pyplot as plt
import seaborn as sns

seed = 55


class Clust_Dataset(Dataset):
    def __init__(self, data):
        self.df = df

    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, idx):
        x = self.data[idx]
        return x


class EncoDeco(nn.Module):
    def __init__(self):
        super(EncoDeco, self).__init__()
        self.input_size = 464 
        self.linear_01 = 150
        self.linear_out = 30
        
        self.enco_lin_01 = nn.Linear(self.input_size, self.linear_01)
        self.enco_lin_03 = nn.Linear(self.linear_01, self.linear_out)        
        
        self.deco_lin_01 = nn.Linear(self.linear_out, self.linear_01)
        self.deco_lin_03 = nn.Linear(self.linear_01, self.input_size)        
        
    def encoder(self, x):
        x = self.enco_lin_01(x)
        x = F.relu(x)
        x = self.enco_lin_03(x)
        return x
    
    def decoder(self, x):
        x = self.deco_lin_01(x)
        x = F.relu(x)
        x = self.deco_lin_03(x)
        x = F.relu(x)
        return x
        
    def forward(self, x):
        encoded = self.encoder(x)
        x_gen = self.decoder(encoded)
        return encoded, F.log_softmax(x_gen, -1)
    
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)


def DEC_loss_func(feat):
    feat = feat.to(device)
    q_numerator = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2))
    q_denominator = 1.0 / torch.sum(q_numerator, dim=1) 
    q = torch.transpose(q_numerator, 0, 1) * q_denominator
    q = torch.transpose(q, 0, 1)
    
    p_numerator = q ** 2 /torch.sum(q, dim=0)
    p_denominator = torch.sum(p_numerator, dim=1)
    p_numerator = torch.transpose(p_numerator, 0, 1)    
    p = p_numerator/p_denominator
    p = torch.transpose(p, 0, 1)
    
    log_q = torch.log(q)
    loss = F.kl_div(log_q, p, reduction='batchmean')
    return loss, p

if __name__=="__main__":
    # read data here
    with open('data.pickle', 'rb') as pp:
        data = pickle.load(data)

    # load pre-trained encoder-decoder
    endn_dense = EncoDeco().to(device)
    endn_dense.load_state_dict(torch.load('saved_model.pt'))

    numClust = 20
    dataset_all = Clust_Dataset(df)
    data_loader = DataLoader(dataset_all, batch_size=2048, num_workers=1, pin_memory=True)

    x_init = data[:100].cuda()
    feat_init, _ = endn_dense(x_init)

    kmeans = KMeans(n_clusters=numClust, n_init=7)
    y_pred_init = kmeans.fit_predict(feat_init.cpu().detach().numpy())
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor).cuda().requires_grad_()


    ClustOptim = torch.optim.SGD(list(endn_dense.parameters()) + [cluster_centers], lr=0.05)
    endn_dense.train()
    silScore = []
    decLoss = []
    for epoch in range(20):
        for X in data_loader:
            X = X.to(device)

            ClustOptim.zero_grad()

            clust_feat, _ = endn_dense(X)
            DEC_loss, _ = DEC_loss_func(clust_feat)

            DEC_loss.backward()
            ClustOptim.step()

        sample_feat = clust_feat.cpu().detach().numpy()
        sample_kmeans = kmeans.predict(clust_feat.cpu().detach().numpy())

        silScore.append(silhouette_score(sample_feat, sample_kmeans)) 
        decLoss.append(DEC_loss.detach().item())

    print(' epoch', '%03d' %epoch, 
          ' loss:', '%.6f' %(DEC_loss.detach().item()*100),
          ' SilScore:', '%.6f' %silhouette_score(sample_feat, sample_kmeans))


    with torch.no_grad():
        clust_feat = []
        for px, user in data_loader:
            px = px.to(device)
            px_h = endn_dense.encoder(px)
            clust_feat.append(px_h)
    clust_feat_out = torch.cat(clust_feat)
