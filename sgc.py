import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, SGConv

import numpy as np
import dgl
from dgl import DGLGraph
import networkx as nx
from config import cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        
        self.sgc = SGConv(cfg.fea_size,
                   cfg.h_size,
                   k=cfg.layers,
                   cached=True,
                   bias=cfg.bias)
       

    def forward(self, g, inputs, calc_mad=False):
        h = self.sgc(g, inputs)
        x = 0
        return h,x



