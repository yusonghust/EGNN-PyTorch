import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GraphConv
import networkx as nx
from config import cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(cfg.layers):
            if i==0:
                indim = cfg.fea_size
            else:
                indim = cfg.h_size
            odim = cfg.h_size
            self.layers.append(GraphConv(indim,odim,activation=cfg.act))
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self,g,features,calc_mad=False):
        x = features
        outputs = []
        for i in range(cfg.layers):
            x = self.dropout(x)
            x = self.layers[i](g,x)
            if calc_mad:
                outputs.append(x)
        return x,outputs