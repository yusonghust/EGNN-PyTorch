import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv

import numpy as np

import networkx as nx
from config import cfg
import os



class GNN(nn.Module):
    def __init__(self,
                 ):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        
        in_feats = cfg.fea_size
        n_hidden = cfg.h_size
        n_classes = cfg.num_class
        n_layers = cfg.layers
        activation = cfg.act 
        dropout = cfg.drop_rate
        aggregator_type = cfg.agg_type
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        #self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, g, features, calc_mad=False):
        h = features
        outputs = []
        for layer in self.layers:
            h = layer(g, h)
            if calc_mad:
                outputs.append(h)
        return h,outputs
